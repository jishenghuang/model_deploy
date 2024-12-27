import ctypes
import os
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
              'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def build_engine_onnx(model_path, trt_path, min_shape = (1, 3, 640, 640),opt_shape = (1, 3, 640, 640),max_shape = (1, 3, 640, 640)):
    logger = trt.Logger(trt.Logger.WARNING)
    if not os.path.exists(trt_path):
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        # config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
        parser = trt.OnnxParser(network, logger)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # Create optimization profile for dynamic input shapes
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        engine_bytes = builder.build_serialized_network(network, config)
        # 保存引擎文件
        with open(trt_path, "wb") as f:
            f.write(engine_bytes)
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_bytes)
    else:
        with open(trt_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

def inference(engine, input_data):
    context = engine.create_execution_context()
    # 初始化流和绑定
    stream = cudart.cudaStreamCreate()[1]
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 检查输入数据形状是否与 profile 匹配
    input_shape = input_data.shape
    context.set_input_shape(input_name, input_shape)   # 设置实际输入的形状
    
    # 分配内存
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    input_size = np.prod(input_shape)
    output_size = np.prod(output_shape)
    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))

    d_input = cudart.cudaMalloc(input_size * input_dtype.itemsize)[1]
    d_output = cudart.cudaMalloc(output_size * output_dtype.itemsize)[1]

    # 拷贝输入数据到设备
    cudart.cudaMemcpy(d_input, input_data.ctypes.data, input_size * input_dtype.itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    
    # 设置上下文绑定
    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)

    # 执行推理
    context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    # 拷贝输出数据回主机
    output_data = np.empty(output_shape, dtype=output_dtype)
    cudart.cudaMemcpy(output_data.ctypes.data, d_output, output_size * output_dtype.itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # 清理资源
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return output_data
def pre_process(image_path, input_size=640):
    # 1. 读取图像
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]  # 保存原始尺寸 (H, W)
    # 2. 调整尺寸并保持宽高比
    scale = input_size / max(original_shape)
    resized_image = cv2.resize(image, (int(original_shape[1] * scale), int(original_shape[0] * scale)))
    # 计算填充的大小
    pad_x = (input_size - resized_image.shape[1]) // 2
    pad_y = (input_size - resized_image.shape[0]) // 2
    # 创建填充图像，大小为 input_size x input_size，初始值为 0（黑色）
    padded_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    # 将调整大小后的图像粘贴到填充图像的中心
    padded_image[pad_y:pad_y + resized_image.shape[0], pad_x:pad_x + resized_image.shape[1]] = resized_image

    # 3. 转为 Float32 格式并归一化
    normalized_image = padded_image.astype(np.float32) / 255.0

    # 4. 转为 NCHW 格式 (1, 3, H, W)
    input_tensor = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, ...]

    return input_tensor, scale, pad_x, pad_y
def post_process(output, scale, pad_x, pad_y, input_size=640, conf_threshold=0.5, iou_threshold=0.4):
    # 1. 解析模型输出 (YOLOv8 通常有一个输出)
    predictions = output  # (batch, num_boxes, 6)
    predictions = predictions[0]  # 获取第一个 batch
    predictions = np.transpose(predictions, (1, 0))
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        confidence = np.max(pred[4:])  # 第5列向后为80个类别概率，取最大的作为置信度
        id = np.argmax(pred[4:])
        if confidence >= conf_threshold:
            x_center, y_center, width, height = pred[:4]
            # 修正边界框坐标，考虑padding
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            width = width / scale
            height = height / scale
            
            # 计算边界框的最小和最大坐标
            x_min = (x_center - width / 2) 
            y_min = (y_center - height / 2)
            x_max = (x_center + width / 2)
            y_max = (y_center + height / 2)
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(confidence)
            class_ids.append(int(id))

    # 2. NMS (非极大值抑制)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    result_boxes = [boxes[i] for i in indices]
    result_scores = [scores[i] for i in indices]
    result_class_ids = [class_ids[i] for i in indices]

    return result_boxes, result_scores, result_class_ids
def draw_boxes(boxes, scores, class_ids, image_path, class_names):
    image = cv2.imread(image_path)
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite("data/result_bus.jpg", image)
    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__== "__main__":
    imgpath = "model_deploy/data/bus.jpg"
    input_tensor, scale, pad_x, pad_y = pre_process(imgpath)
    engine = build_engine_onnx("models/yolov8s.onnx",trt_path="models/yolov8s.trt")
    output = inference(engine, input_tensor)
    result = post_process(output, scale, pad_x, pad_y)
    draw_boxes(*result, image_path=imgpath, class_names=class_names)

