当然可以！我们来实现一个**简单的注意力机制（Attention Mechanism）**，比如**加性注意力（Additive Attention）**，也称为 **Bahdanau Attention**，它最初用于机器翻译任务中的序列到序列模型（Sequence-to-Sequence），能帮助模型关注输入序列中更相关的部分。

---

## 🧠 注意力机制简介

注意力机制允许模型在处理当前输出时，动态地关注输入序列中不同的位置。常见的注意力类型包括：

| 类型 | 特点 |
|------|------|
| 加性注意力（Additive / Bahdanau） | 使用一个可学习的隐藏层来计算注意力得分 |
| 点积注意力（Dot-Product） | 简单计算 query 和 key 的点积 |
| 缩放点积注意力（Scaled Dot-Product） | Transformer 中使用，点积后除以 $\sqrt{d_k}$ |

---

## ✅ 加性注意力（Additive Attention）公式

给定：

- **query**：当前解码器状态（如：$ h_t $）
- **keys**：所有编码器状态（如：$ h_1, h_2, ..., h_T $）

计算注意力得分：

$$
e_i = v^T \tanh(W h_i + b)
$$

然后使用 softmax 归一化：

$$
\alpha_i = \frac{e^{e_i}}{\sum_j e^{e_j}}
$$

最后加权求和得到上下文向量：

$$
c = \sum_i \alpha_i h_i
$$

---

## 💻 PyTorch 实现加性注意力层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys):
        """
        Args:
            query: [batch_size, hidden_dim]，当前解码器状态
            keys:  [batch_size, seq_len, hidden_dim]，所有编码器状态

        Returns:
            context: [batch_size, hidden_dim]，加权上下文向量
            weights: [batch_size, seq_len]，注意力权重分布
        """
        # Step 1: 计算 W * h_i
        energy = self.W(keys)  # shape: [batch_size, seq_len, hidden_dim]

        # Step 2: 加上 query（广播机制）
        # query.unsqueeze(1): [batch_size, 1, hidden_dim]
        # energy: [batch_size, seq_len, hidden_dim]
        energy = torch.tanh(energy + query.unsqueeze(1))

        # Step 3: 计算 v^T * tanh(...)
        energy = self.v(energy).squeeze(-1)  # shape: [batch_size, seq_len]

        # Step 4: softmax 得到注意力权重
        weights = F.softmax(energy, dim=1)  # shape: [batch_size, seq_len]

        # Step 5: 加权求和
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # shape: [batch_size, hidden_dim]

        return context, weights
```

---

## 🧪 示例用法

```python
# 假设 batch_size=2, seq_len=5, hidden_dim=10
batch_size = 2
seq_len = 5
hidden_dim = 10

# 随机生成 query 和 keys
query = torch.randn(batch_size, hidden_dim)  # 当前解码器状态
keys = torch.randn(batch_size, seq_len, hidden_dim)  # 所有编码器状态

# 实例化注意力层
attention = AdditiveAttention(hidden_dim)

# 前向传播
context, weights = attention(query, keys)

print("上下文向量 shape:", context.shape)  # [2, 10]
print("注意力权重 shape:", weights.shape)  # [2, 5]
print("注意力权重示例:\n", weights)
```

---

## 📈 注意力权重可视化示例（可选）

```python
import matplotlib.pyplot as plt

# 画出第一个样本的注意力权重
plt.bar(range(seq_len), weights[0].detach().numpy())
plt.xlabel("输入位置")
plt.ylabel("注意力权重")
plt.title("Additive Attention Weights")
plt.show()
```

---

## ✅ 总结对比表

| 方法 | 公式 | 适用场景 | 是否可微 |
|------|------|----------|-----------|
| 加性注意力 | $ v^T \tanh(W h_i + b) $ | 序列建模、RNN-based 模型 | ✅ 是 |
| 点积注意力 | $ q \cdot k_i $ | 快速计算，维度一致时 | ✅ 是 |
| 缩放点积注意力 | $ \frac{q \cdot k_i}{\sqrt{d_k}} $ | Transformer 等深层模型 | ✅ 是 |

---

## 🧩 进一步扩展建议

你可以继续：

- 把注意力机制集成到 Seq2Seq 模型中（如 Encoder-Decoder）
- 实现 **点积注意力** 或 **多头注意力（Multi-head Attention）**
- 使用 `nn.MultiheadAttention` 模块
- 在机器翻译、文本摘要等任务中应用注意力机制
- 使用 `torch.nn.utils.rnn.pack_padded_sequence` 处理变长序列

---