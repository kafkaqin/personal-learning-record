非常好的任务！我们将使用 **PyTorch** 实现一个 **简单的生成对抗网络（GAN）**，用于生成 **MNIST 手写数字图像**。

---

## 🧠 GAN 简介

GAN 由两个网络组成：

| 网络 | 功能 |
|------|------|
| **生成器（Generator）** | 从随机噪声生成图像 |
| **判别器（Discriminator）** | 判断图像是真实还是生成的 |

训练过程是一个 **零和博弈（zero-sum game）**，目标函数为：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

---

## ✅ 本例目标

我们将实现：

- 一个简单的 **全连接 GAN（DCGAN 的简化版）**
- 使用 **MNIST 手写数字数据集**
- 使用 **Binary Cross Entropy Loss（BCELoss）**
- 使用 **Adam 优化器**
- 每隔几个 epoch 可视化生成的图像

---

## 📦 所需库

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
```

---

## 🧱 定义 Generator（生成器）

```python
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)
```

---

## 🧱 定义 Discriminator（判别器）

```python
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        return self.net(x)
```

---

## 📥 加载 MNIST 数据集

```python
# 数据预处理：归一化到 [-1, 1]，因为 Generator 输出是 Tanh
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
```

---

## 🚀 初始化模型、损失函数和优化器

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
latent_dim = 100
lr = 0.0002
epochs = 100

# 初始化模型
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 损失函数
criterion = nn.BCELoss()

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
```

---

## 📈 可视化生成图像函数

```python
def visualize_images(generator, device, epoch):
    z = torch.randn(16, latent_dim).to(device)
    with torch.no_grad():
        generated = generator(z).cpu().view(-1, 28, 28)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.close()
```

---

## 🏁 训练循环

```python
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  训练判别器 D
        # ---------------------
        # 真实图像的损失
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        # 生成器生成假图像
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  训练生成器 G
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # 每 10 个 epoch 可视化一次
    if (epoch + 1) % 10 == 0:
        visualize_images(generator, device, epoch + 1)
```

---

## 📈 示例输出（可能略有不同）

```
Epoch [1/100] Loss D: 1.3689, Loss G: 0.7231
Epoch [2/100] Loss D: 1.1452, Loss G: 0.6213
...
Epoch [100/100] Loss D: 0.3214, Loss G: 0.6901
```

可视化图像会保存为 `generated_epoch_10.png`、`generated_epoch_20.png` 等。

---

## ✅ 总结对比表

| 组件 | 说明 |
|------|------|
| Generator | 从随机噪声生成图像（784维） |
| Discriminator | 判断图像是否为真 |
| 损失函数 | BCE Loss（Binary Cross Entropy） |
| 优化器 | Adam（学习率 0.0002） |
| 效果 | 50 个 epoch 后可生成较清晰的数字图像 |

---

## 🧩 进一步扩展建议

你可以继续：

- 使用 **DCGAN（深度卷积 GAN）** 提升图像质量
- 使用 **Wasserstein GAN（WGAN）** 改进训练稳定性
- 使用 **LSGAN（最小二乘 GAN）** 替代 BCE Loss
- 添加 **注意力机制** 到生成器或判别器
- 将 GAN 应用于 **图像到图像翻译（如 Pix2Pix）**

---