当然可以！下面是一个使用 **NumPy** 实现的 **简单全连接神经网络（Fully Connected Neural Network）的前向传播过程**，适用于多分类任务。

我们将实现一个两层神经网络（输入层 → 隐藏层 → 输出层）：

---

## 🧠 一、模型结构

- 输入层：大小 `input_size`（例如特征维度）
- 隐藏层：大小 `hidden_size`，使用 **ReLU 激活函数**
- 输出层：大小 `output_size`，使用 **Softmax 激活函数**
- 损失函数：**交叉熵损失（Cross-Entropy Loss）**

---

## ✅ 二、Python 示例代码（NumPy）

```python
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 模拟数据
num_samples = 100
input_size = 4
hidden_size = 10
output_size = 3

# 随机生成一些输入数据和 one-hot 编码的标签
X = np.random.randn(num_samples, input_size)
y_true = np.eye(output_size)[np.random.choice(output_size, num_samples)]

# 初始化参数
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# 定义激活函数
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性处理
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 前向传播
def forward(X, W1, b1, W2, b2):
    # 第一层线性变换 + ReLU 激活
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # 第二层线性变换 + Softmax 输出
    z2 = a1 @ W2 + b2
    y_pred = softmax(z2)

    return y_pred, a1, z1

# 执行前向传播
y_pred, a1, z1 = forward(X, W1, b1, W2, b2)

# 查看输出形状
print("输入 X 形状:", X.shape)
print("预测概率 y_pred 形状:", y_pred.shape)
print("隐藏层激活值 a1 形状:", a1.shape)
```

---

## ✅ 输出示例：

```
输入 X 形状: (100, 4)
预测概率 y_pred 形状: (100, 3)
隐藏层激活值 a1 形状: (100, 10)
```

说明：
- 每个样本有 4 个特征，经过隐藏层后变为 10 维。
- 最终输出是 3 类的概率分布（Softmax 输出）。

---

## 📌 三、关键公式总结

| 层级 | 公式 |
|------|------|
| 隐藏层（ReLU） | $ a^{(1)} = \text{ReLU}(X W^{(1)} + b^{(1)}) $ |
| 输出层（Softmax） | $ \hat{y} = \text{Softmax}(a^{(1)} W^{(2)} + b^{(2)}) $ |

---

## 🧪 四、后续扩展建议

你可以在此基础上继续添加以下功能：

| 功能 | 实现方式 |
|------|----------|
| 反向传播 | 计算梯度并更新参数 |
| 损失计算 | 使用交叉熵损失函数 |
| 多轮训练 | 添加 for 循环进行迭代优化 |
| 使用自动求导 | 改为 PyTorch 或 JAX 版本 |
| 加入正则化 | 在损失中加入 L2 正则项 |

---

## 📊 五、完整训练流程（简要示意）

如果你想继续实现反向传播和参数更新，这里是简化的步骤：

```python
learning_rate = 1e-3

for epoch in range(100):
    # 前向传播
    y_pred, a1, z1 = forward(X, W1, b1, W2, b2)

    # 计算损失（略）

    # 反向传播
    dy = y_pred - y_true
    dW2 = a1.T @ dy
    db2 = np.sum(dy, axis=0, keepdims=True)

    da1 = dy @ W2.T
    dz1 = da1 * (z1 > 0)  # ReLU 导数
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # 参数更新
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
```

---

