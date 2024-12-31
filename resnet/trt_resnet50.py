import os
import sys
sys.path.append(os.getcwd())
import time
import tensorrt as trt
from tqdm import tqdm
from model_deploy.resnet.quent_resnet import build_dataloader
from model_deploy.resnet.export_resnet50 import build_trt
print("trt Version: ", trt.__version__)
import onnxruntime as ort
import numpy as np
from cuda import cudart

def build_dataset():
    np.random.seed(1)
    num_samples = 10000
    data = [np.random.randn(1, 3, 256, 256).astype(np.float32) for _ in range(num_samples)]  # 使用列表生成式
    # 访问数据集中的第一个样本
    sample = data[0]
    print(sample.shape)  # 输出 (1, 3, 256, 256)
    print(sample)
    return data
def inference_trt(engine, input_data):
    input_data = np.ascontiguousarray(input_data)
    context = engine.create_execution_context()
    
    # 初始化流和绑定
    stream = cudart.cudaStreamCreate()[1]
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # # 检查输入数据形状是否与 profile 匹配
    ac_input_shape = input_data.shape
    context.set_input_shape(input_name, ac_input_shape)   # 设置实际输入的形状
    # 分配内存
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    input_shape[0] = ac_input_shape[0]
    output_shape[0] = ac_input_shape[0]
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
if __name__ == "__main__":
    engine = build_trt("models/mse_resnet50_dynamic.onnx",trt_path="models/mse_resnet50_dynamic.trt",use_int8=True)
    # data = build_dataset()
    data_loader = build_dataloader()
    # 获取批量数据
    since = time.time()
    for i, (image, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if i >= len(data_loader):
            break
        inference_trt(engine,image)
    # for i in range(0, len(data), batch_size):
    #     batch = data[i:i+batch_size]  # 获取一个批次
    #     batch_data = np.concatenate(batch, axis=0)  # 将批次数据沿第一维连接，形状为 [batch_size, 3, 256, 256]
    #     # print(batch_data.shape)  # 输出形状 (1, 3, 256, 256)
    #     outputs = inference_trt(engine,batch_data)
    #     # print(outputs.shape)
    # print(batch_data.shape)
    # print(outputs.shape)
    time_elapsed = time.time() - since
    print(f'inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        


