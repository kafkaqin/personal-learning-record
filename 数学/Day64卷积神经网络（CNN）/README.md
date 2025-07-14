当然可以！下面是一个使用 **PyTorch** 实现的简单 **卷积神经网络（CNN）**，用于对 **MNIST 手写数字数据集** 进行分类任务。

---

## 🧠 模型结构概述

我们将构建一个简单的 CNN，包含：

1. 一个卷积层 `Conv2d`
2. 一个最大池化层 `MaxPool2d`
3. 一个激活函数 `ReLU`
4. 全连接层（Linear）
5. 使用 `CrossEntropyLoss` 损失函数
6. 使用 `Adam` 优化器

---

## ✅ 数据集介绍

- **MNIST**：手写数字图像数据集（0~9），图像大小为 28x28，单通道灰度图
- 包含 60,000 张训练图像，10,000 张测试图像

---

## 📦 所需库

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```

---

## 🧱 模型定义

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # (28x28 -> 28x28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28x28 -> 14x14)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 📥 数据加载与预处理

```python
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
```

---

## 🚀 训练函数

```python
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f'\tLoss: {loss.item():.6f}')
```

---

## 🧪 测试函数

```python
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
```

---

## 🏁 主函数运行训练

```python
def main():
    # 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型、损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练多个 epoch
    for epoch in range(1, 6):  # 训练5个epoch
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
```

---

## 📈 示例输出（每次运行略有不同）

```
Train Epoch: 1 [0/60000]	Loss: 2.310129
Train Epoch: 1 [6400/60000]	Loss: 0.167452
...
Test set: Average loss: 0.0472, Accuracy: 9850/10000 (98.50%)
```

---

## ✅ 总结

我们完成了以下内容：

| 功能 | 说明 |
|------|------|
| 模型结构 | CNN（卷积 + ReLU + MaxPool + 全连接） |
| 数据集 | MNIST 手写数字（10类） |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam |
| 设备支持 | CPU / GPU |
| 准确率 | 可达 98%+ |

---

## 🧩 后续扩展建议

你可以继续：

- 添加更多卷积层或使用 `nn.Sequential` 构建更复杂的模型
- 使用 `torchvision.models` 加载预训练模型
- 添加可视化（使用 `matplotlib` 显示预测结果）
- 使用 TensorBoard 可视化训练过程
