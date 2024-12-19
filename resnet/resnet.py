import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pytorch_quantization import quant_modules
from torch import optim
from tqdm import tqdm
# from train import evaluate, train_one_epoch, load_data
class BasicBlock(nn.Module):
    """
    ResNet 的基础残差模块，用于 ResNet-18 和 ResNet-34。
    """
    expansion = 1  # 输出通道倍数（对 ResNet-18 来说保持不变）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化基础块。
        :param in_channels: 输入通道数。
        :param out_channels: 输出通道数。
        :param stride: 步长，默认值为 1。
        :param downsample: 下采样层（可选）。
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于匹配残差的尺寸和通道

    def forward(self, x):
        identity = x  # 保留输入以便残差连接
        if self.downsample is not None:
            identity = self.downsample(x)  # 调整残差形状以匹配输出
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 添加残差
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet 通用类，用于构建 ResNet-18 和其他变种。
    """
    def __init__(self, block, layers, num_classes=1000):
        """
        初始化 ResNet。
        :param block: 残差块类型。
        :param layers: 每个阶段的残差块数量。
        :param num_classes: 输出类别数。
        """
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始输入通道数

        # 第一层：卷积 + 批归一化 + 激活 + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 主体
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建 ResNet 的层级。
        :param block: 残差块类型。
        :param out_channels: 输出通道数。
        :param num_blocks: 残差块数量。
        :param stride: 第一个块的步长。
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 下采样模块用于调整输入形状
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 构建 ResNet-18
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
def preprocess(image):
    # 定义图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor
def build_dataloaders():
    seed = 1
    # 设置好读取数据的目录
    data_dir = './model_deploy/datasets'
    train_dir = 'mam'
    valid_dir = 'val'
    '''
    制作好数据源：
        data_transforms中指定了所有图像预处理操作（数据增强）
        ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
    '''
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

    #指定batch_size 为128  也即每次从集合中拿128个样本点进行训练或者测试
    batch_size = 32
    # 加载完整的数据集（不带变换）
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, train_dir))
    # 提取样本索引和标签
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.targets[i] for i in indices]
    # 按类别分层划分训练集和测试集
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=seed
    )
    # 定义子集数据集，分别应用不同的数据增强
    train_dataset = Subset(full_dataset, train_indices)
    train_dataset.dataset = datasets.ImageFolder(
        os.path.join(data_dir, train_dir), transform=data_transforms["train"]
    )
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset = datasets.ImageFolder(
        os.path.join(data_dir, train_dir), transform=data_transforms["val"]
    )
    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 检查样本数量
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    # # 检查类别分布
    # check_class_distribution(train_dataset, "训练集")
    # check_class_distribution(test_dataset, "测试集")
    # # 检查训练集和测试集样本
    # visualize_samples(train_loader, "训练集样本")
    # visualize_samples(test_loader, "测试集样本")
    return {"train":train_loader, "val":test_loader}
def check_class_distribution(dataset, name):
    # 检查类别分布
    from collections import Counter
    labels = [dataset.dataset.targets[i] for i in dataset.indices]
    class_count = dict(sorted(Counter(labels).items()))
    print(f"{name} 类别分布: {class_count}")
# 可视化数据加载器样本
def visualize_samples(loader, title):
    import matplotlib.pyplot as plt
    images, labels = next(iter(loader))
    plt.figure(figsize=(8, 8))
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()
        plt.subplot(2, 2, i+1)
        plt.title(f"{title} - 类别: {labels[i].item()}")
        plt.imshow(img)
    plt.show()
def quent_model():
    quant_modules.initialize()
    # 定义需要替换的浮点模块列表
    # float_module_list = ["Linear"]
    # 自定义量化模块映射列表
    # custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]
    # 初始化量化模块
    # quant_modules.initialize(float_module_list, custom_quant_modules)
    # quant_desc_input = QuantDescriptor(calib_method='histogram')
    # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
def build_model(num_classes = 1000):
    model = resnet18(num_classes=45)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    print(model)
    return model
def build_criterion():
    return nn.CrossEntropyLoss()
def build_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
def train_model(dataloaders,model,criterion,optimizer,num_epochs=25,device="cuda"):
    model.to(device)
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        # 每个 epoch 有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段进行反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model
def eval_model(model, dataloaders, criterion, device="cuda"):
    model.to(device)
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # 遍历验证集
    with torch.no_grad():  # 禁用梯度计算，以节省内存和计算
        for inputs, labels in tqdm(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    # 计算平均损失和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    print(f'Evaluation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc
def save_model(model):
    torch.save(model.state_dict(), 'model_deploy/resnet/checkpoint/best_model.pth')
def load_model(model_path='model_deploy/resnet/checkpoint/best_model.pth'):
    model       = build_model(num_classes=45)
    # 加载保存的权重
    model.load_state_dict(torch.load(model_path))
    return model
if __name__ == "__main__":
    quent_model()
    dataloaders = build_dataloaders()
    model       = build_model(num_classes=45)
    criterion   = build_criterion()
    optimizer   = build_optimizer(model)
    model       = train_model(dataloaders,model,criterion,optimizer,num_epochs=5)
    save_model(model)
    # model = load_model()
    eval_loss, eval_acc = eval_model(model,dataloaders,criterion)
