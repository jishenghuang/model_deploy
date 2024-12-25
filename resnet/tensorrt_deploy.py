import ctypes
import numpy as np
import tensorrt as trt
from cuda import cudart
def build_engine_onnx(model_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    # 保存引擎文件
    with open("yolo11n.trt", "wb") as f:
        f.write(engine_bytes)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(engine_bytes)

def inference(engine, context, input_data):
    # 初始化流和绑定
    stream = cudart.cudaStreamCreate()[1]
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
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
build_engine_onnx("yolo11n.onnx")