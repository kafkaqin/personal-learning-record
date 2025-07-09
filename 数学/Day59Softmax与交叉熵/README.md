
---

## 🧠 一、Softmax 函数简介

Softmax 函数将一个实数向量转换为概率分布：

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
$$

其中 $ C $ 是类别总数。输出值都在 [0,1] 区间，并且总和为 1，因此可以看作是每个类别的预测概率。

---

## 🧠 二、交叉熵损失函数简介

对于真实标签 $ y \in \{0, 1\}^C $ 和预测概率 $ \hat{y} \in (0,1)^C $，交叉熵定义为：

$$
\mathcal{L} = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$

这是多分类模型中最常用的损失函数之一。

---

## ✅ 三、Python 实现代码（NumPy）

我们将在 NumPy 中手动实现 Softmax 和交叉熵损失函数，并结合一个简单的分类任务进行演示。

```python
import numpy as np

# 1. 定义 Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止数值溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 2. 定义交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-15)
    loss = np.mean(log_likelihood)
    return loss

# 3. 模拟数据：假设我们有3个类别，每条数据有4个特征
np.random.seed(0)
X = np.random.randn(6, 4)  # 6个样本，4维特征

# 真实标签（one-hot 编码）
y_true = np.array([
    [1, 0, 0],  # 类别0
    [0, 1, 0],  # 类别1
    [0, 0, 1],  # 类别2
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# 初始化权重和偏置项
W = np.random.randn(4, 3) * 0.01  # 输入维度:4, 输出维度:3
b = np.zeros((1, 3))

# 4. 前向传播：计算 logits 和概率
logits = X @ W + b
y_pred = softmax(logits)

# 5. 计算损失
loss = cross_entropy_loss(y_true, y_pred)
print("交叉熵损失:", loss)
```

---

## ✅ 输出示例：

```
交叉熵损失: 1.118279459912966
```

---

## 📌 四、梯度下降更新参数（可选）

我们可以继续添加反向传播来优化权重：

```python
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    # 前向传播
    logits = X @ W + b
    y_pred = softmax(logits)

    # 损失
    loss = cross_entropy_loss(y_true, y_pred)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # 反向传播
    grad_logits = y_pred - y_true  # 简化后的梯度
    grad_W = X.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0, keepdims=True)

    # 参数更新
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b
```

---

## ✅ 最终输出示例：

```
Epoch 0: Loss = 1.1183
Epoch 100: Loss = 0.3512
Epoch 200: Loss = 0.1649
...
Epoch 900: Loss = 0.0041
```

说明模型正在逐步收敛，能够更准确地预测类别概率。

---

## 📊 五、关键公式总结

| 公式 | 描述 |
|------|------|
| $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | 将输出转为概率分布 |
|$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N y_i \log(\hat{y}_i)$| 交叉熵损失 |
| $\nabla_{\text{logits}} \mathcal{L} = \hat{y} - y$ | Softmax + Cross Entropy 的梯度简化形式 |

---

## 🧪 六、扩展建议

| 功能 | 方法 |
|------|------|
| 多层网络 | 使用全连接层 + ReLU 激活 |
| 自动求导 | 改用 PyTorch 或 JAX 实现 |
| 批量训练 | 使用 `DataLoader` 或手动分 batch |
| 正则化 | 在损失中加入 L2 正则项 |
| 多分类评估 | 使用 `accuracy_score`, `confusion_matrix` 等指标 |

---

如果你希望：
- 使用 PyTorch 实现相同功能
- 加入正则化或批量归一化
- 构建完整的神经网络模型
