
import os
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
import torch
import torchvision
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from torchvision import models
from torchvision import transforms, models, datasets
from tqdm import tqdm

def build_dataloader():
    data_transforms = {
        'train': 
            transforms.Compose([
            transforms.Resize([256, 256]), #设置好每张图片大小相同
            transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
            # transforms.CenterCrop(64),#将图片随机裁剪为64x64
            transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(), #由于PyTorch框架数据格式必须为Tensor，因此转换数据为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#对三通道分别做均值，标准差
        ]),
        'val': 
            transforms.Compose([
            transforms.Resize([256, 256]), #必须与训练集中transforms.CenterCrop(64)大小相同
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_path = "datasets"
    batch_size = 32
    dir = os.path.join(data_path, 'mam')

    dataset = datasets.ImageFolder(dir, transform=data_transforms["val"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()
if __name__ == "__main__":
    quant_modules.initialize()
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model = models.resnet50(pretrained=True)
    model.cuda()
    dataloader = build_dataloader()
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, dataloader, num_batches=40)
        compute_amax(model, method="percentile", percentile=99.99)
    print(model)
    torch.save(model.state_dict(), "model_deploy/resnet/checkpoint/quant_resnet50-calibrated.pth")

