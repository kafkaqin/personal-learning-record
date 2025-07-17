当然可以！我们来演示如何在 **PyTorch** 中：

- 使用内置的 **损失函数**（如 `MSELoss`, `CrossEntropyLoss`）
- 对模型参数进行 **前向传播**
- 使用 `loss.backward()` 自动计算梯度
- 查看参数的 `.grad` 属性来观察梯度值

---

## 🧠 本例目标

我们将：

1. 定义一个简单的线性模型（如 `y = Wx + b`）
2. 使用 PyTorch 的损失函数（如 `MSELoss`）
3. 手动传入输入和目标
4. 计算损失并反向传播
5. 查看参数的梯度

---

## ✅ 示例代码（PyTorch 中使用损失函数并计算梯度）

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以确保可复现
torch.manual_seed(42)

# -------------------------------------
# 1. 定义一个简单的线性模型
# -------------------------------------
# y = Wx + b，其中 W 是权重，b 是偏置
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  # 简单线性层

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# 查看模型参数
print("初始模型参数：")
print(model.state_dict())
```

---

### 🔍 模型结构说明：

```python
SimpleModel(
  (linear): Linear(in_features=1, out_features=1, bias=True)
)
```

- `in_features=1`：表示输入特征维度为 1
- `out_features=1`：输出维度也为 1
- `bias=True`：模型包含偏置项

---

```python
# -------------------------------------
# 2. 定义损失函数（如 MSE Loss）
# -------------------------------------
criterion = nn.MSELoss()

# -------------------------------------
# 3. 定义优化器（如 SGD 或 Adam）
# -------------------------------------
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------------------------
# 4. 构造输入和目标
# -------------------------------------
# 假设我们有 3 个样本，每个样本一个特征
inputs = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
targets = torch.tensor([[2.0], [4.0], [6.0]])  # 目标是 2x

# -------------------------------------
# 5. 前向传播
# -------------------------------------
outputs = model(inputs)
loss = criterion(outputs, targets)

print("\n前向传播输出：")
print(outputs)
print("\n计算损失值：", loss.item())

# -------------------------------------
# 6. 反向传播计算梯度
# -------------------------------------
optimizer.zero_grad()  # 清空之前的梯度
loss.backward()        # 反向传播计算梯度

# -------------------------------------
# 7. 查看参数梯度
# -------------------------------------
print("\n参数梯度：")
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
```

---

## 📈 示例输出（可能略有不同）

```
初始模型参数：
OrderedDict([('linear.weight', tensor([[0.8200]])), ('linear.bias', tensor([0.]))])

前向传播输出：
tensor([[0.8200],
        [1.6400],
        [2.4600]], grad_fn=<AddmmBackward>)

计算损失值： 5.521199703216553

参数梯度：
linear.weight: tensor([[-4.9200]])
linear.bias: tensor([-3.0000])
```

---

## 📌 关键点说明

| 步骤 | 说明 |
|------|------|
| `loss.backward()` | 计算所有参数的梯度（基于当前损失） |
| `param.grad` | 参数的 `.grad` 属性保存了梯度值 |
| `optimizer.zero_grad()` | 清空梯度，防止梯度叠加 |
| `requires_grad=True` | 输入是否需要计算梯度（一般对模型参数自动设置） |

---

## ✅ 使用其他损失函数（例如分类任务）

如果你做的是分类任务，可以使用：

```python
criterion = nn.CrossEntropyLoss()
```

它会自动结合 `Softmax` 和 `NLLLoss`，适用于多分类任务。

---

## 🧩 后续扩展建议

你可以继续：

- 将损失函数和梯度计算嵌入训练循环
- 使用 `torchviz` 可视化计算图
- 手动实现梯度裁剪（Gradient Clipping）
- 使用 `torch.nn.utils.clip_grad_norm_` 防止梯度爆炸
