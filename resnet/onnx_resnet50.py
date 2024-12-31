import time
import tensorrt as trt
print("trt Version: ", trt.__version__)
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession("models/mse_resnet50_dynamic.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 获取模型输入名称
input_name = session.get_inputs()[0].name
# 生成 100 个形状为 [1, 3, 256, 256] 的随机 NumPy 数组
np.random.seed(1)
num_samples = 1000
data = [np.random.randn(1, 3, 256, 256).astype(np.float32) for _ in range(num_samples)]  # 使用列表生成式
# 访问数据集中的第一个样本
sample = data[0]
print(sample.shape)  # 输出 (1, 3, 256, 256)
# print(sample)

# 获取批量数据
batch_size = 1
since = time.time()
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]  # 获取一个批次
    batch_data = np.concatenate(batch, axis=0)  # 将批次数据沿第一维连接，形状为 [batch_size, 3, 256, 256]
    # print(batch_data.shape)  # 输出形状 (16, 1, 3, 256, 256)
    outputs = session.run(None, {input_name: batch_data})
    # print(outputs[0].shape)
time_elapsed = time.time() - since
print(f'inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
