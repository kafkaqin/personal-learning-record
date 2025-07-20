当然可以！我们将使用 **PyTorch** 构建一个 **卷积神经网络（CNN）**，并在 **CIFAR-10 数据集** 上进行图像分类任务。

---

## 🧠 任务目标

我们将在 **CIFAR-10 数据集** 上训练一个 CNN 分类器，目标是识别以下 10 类图像：

| 类别 | 描述 |
|------|------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## ✅ 所需库

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

---

## 🧱 构建 CNN 模型（PyTorch）

我们使用一个简单的 CNN 结构，包含几个卷积层、池化层和全连接层：

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入：3通道，输出：32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 下采样

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 📥 加载 CIFAR-10 数据集

```python
# 数据预处理（标准化）
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
```

---

## 🚀 初始化模型、损失函数和优化器

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = SimpleCNN().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 📈 可视化训练图像（可选）

```python
def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取一批训练图像
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 显示图像
imshow(torchvision.utils.make_grid(images[:4]))
print('真实标签:', ' '.join(f'{labels[j]}' for j in range(4)))
```

---

## 🏁 训练循环

```python
# 训练参数
num_epochs = 10

# 用于记录训练过程
train_losses = []

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
```

---

## 📊 可视化训练损失曲线

```python
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🧪 测试模型准确率

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')
```

---

## ✅ 示例输出（可能略有不同）

```
Epoch [1/10], Loss: 1.7982
Epoch [2/10], Loss: 1.3912
...
Epoch [10/10], Loss: 0.6821
测试集准确率: 72.34%
```

---

## ✅ 模型性能总结

| 指标 | 值 |
|------|----|
| 数据集 | CIFAR-10（32x32 RGB 图像） |
| 模型结构 | CNN（3个卷积块 + 2个全连接层） |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam（学习率 0.001） |
| 准确率（10 epochs） | ~70% - 75% |

---

## 🧩 进一步扩展建议

你可以继续：

- 使用 **预训练模型（如 ResNet、VGG）** 提升准确率
- 添加 **数据增强（Data Augmentation）**：如随机裁剪、旋转、翻转
- 使用 **学习率调度器（如 StepLR 或 ReduceLROnPlateau）**
- 使用 **混淆矩阵** 分析分类错误
- 使用 **TensorBoard** 可视化训练过程
- 使用 **混合精度训练（AMP）** 加快训练速度

---