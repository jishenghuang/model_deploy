import os
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
import torch
import torchvision
import torch.nn as nn
import tensorrt as trt
def build_onnx():
    # quant_modules.initialize()
    num_classes = 45
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)                                                             

    # load the calibrated model
    state_dict = torch.load("model_deploy/resnet/checkpoint/best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.cuda()

    dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

    # 假设 model 是量化后的 PyTorch 模型
    torch.onnx.export(model, dummy_input, "models/resnet50_dynamic.onnx", 
                    verbose=True, opset_version=13,
                    input_names=["input"],      # 输入名称
                    output_names=["output"],    # 输出名称
                    dynamic_axes={              # 设置动态维度
                        "input": {0: "batch_size"},  # 输入动态维度
                        "output": {0: "batch_size"}  # 输出动态维度
                    })
def build_trt(model_path, trt_path, use_int8=False, min_shape = (1, 3, 256, 256),opt_shape = (128, 3, 256, 256),max_shape = (256, 3, 256, 256)):
    logger = trt.Logger(trt.Logger.WARNING)
    if not os.path.exists(trt_path):
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        if use_int8 == True:
            config.set_flag(trt.BuilderFlag.INT8)   
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
        if use_int8 == True:
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                if tensor:
                    tensor.dynamic_range=(-127, 127)
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                for j in range(layer.num_outputs):
                    tensor = layer.get_output(j)
                    if tensor:
                        tensor.dynamic_range=(-127, 127)
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
if __name__ == "__main__":
    build_onnx()
    engine = build_trt("models/mse_resnet50_dynamic.onnx",trt_path="models/mse_resnet50_dynamic.trt",use_int8=True)

