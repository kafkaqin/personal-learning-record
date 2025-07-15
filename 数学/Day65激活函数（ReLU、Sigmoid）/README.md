非常好！我们来实现几种常用的**激活函数（Activation Functions）**，包括它们的前向传播和反向传播（梯度），并通过可视化观察它们在不同输入下的输出和梯度变化。

---

## 🧠 常见激活函数及其导数

| 激活函数 | 表达式 $ f(x) $ | 导数 $ f'(x) $ |
|----------|------------------|------------------|
| Sigmoid  | $ \frac{1}{1 + e^{-x}} $ | $ f(x)(1 - f(x)) $ |
| Tanh     | $ \tanh(x) $             | $ 1 - f(x)^2 $      |
| ReLU     | $ \max(0, x) $           | $ 1 \text{ if } x > 0, \text{ else } 0 $ |
| Leaky ReLU | $ \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases} $ | $ \begin{cases} 1 & x > 0 \\ \alpha & x < 0 \end{cases} $ |
| ELU      | $ \begin{cases} x & x \geq 0 \\ \alpha (e^x - 1) & x < 0 \end{cases} $ | $ \begin{cases} 1 & x > 0 \\ \alpha e^x & x < 0 \end{cases} $ |

---

## 💻 实现代码：激活函数及其梯度

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

# 生成输入数据
x = np.linspace(-5, 5, 400)

# 所有激活函数及梯度列表
activations = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", lambda x: leaky_relu(x), lambda x: leaky_relu_derivative(x)),
    ("ELU", lambda x: elu(x), lambda x: elu_derivative(x)),
]
```

---

## 📈 可视化激活函数及其梯度

```python
plt.figure(figsize=(15, 10))

for i, (name, act_fn, grad_fn) in enumerate(activations):
    y = act_fn(x)
    dy = grad_fn(x)

    # 绘制激活函数
    plt.subplot(len(activations), 2, 2*i+1)
    plt.plot(x, y, label=name)
    plt.title(f"{name} Activation")
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Output")

    # 绘制梯度
    plt.subplot(len(activations), 2, 2*i+2)
    plt.plot(x, dy, color='orange', label=f"{name} Gradient")
    plt.title(f"{name} Gradient")
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Derivative")

plt.tight_layout()
plt.show()
```

---

## 📊 示例结果说明

- **Sigmoid**：
    - 输出范围 [0, 1]，适合二分类输出层。
    - 梯度在两端趋近于零 → **梯度消失问题**。

- **Tanh**：
    - 输出范围 [-1, 1]，比 Sigmoid 更中心对称。
    - 同样存在梯度消失问题。

- **ReLU**：
    - 在正区间导数恒为 1，缓解了梯度消失。
    - 缺点是负区间“死亡神经元”（梯度为 0）。

- **Leaky ReLU**：
    - 负值部分有一定梯度（默认 α=0.01），缓解 ReLU 的“死区”。

- **ELU**：
    - 对负值也有非零梯度，输出均值接近 0，收敛更快。

---

## ✅ 总结对比表

| 激活函数 | 是否可微 | 梯度是否恒定 | 是否解决梯度消失 | 是否常用 |
|---------|-----------|---------------|-------------------|------------|
| Sigmoid | ✅        | ❌            | ❌                | ⚠️ 一般用于输出层 |
| Tanh    | ✅        | ❌            | ❌                | ⚠️ 曾经流行，现在较少使用 |
| ReLU    | ❌（在0不可导） | ✅（>0时） | ✅                | ✅ 最常用 |
| Leaky ReLU | ❌（在0不可导） | ✅（接近1） | ✅               | ✅ 常用改进版 |
| ELU     | ✅        | ✅（接近1）   | ✅                | ✅ 特定场景下使用 |

---

## 🧩 进一步实验建议

你可以尝试：

- 将这些激活函数应用到你之前实现的 CNN 或全连接网络中
- 使用 PyTorch 中的 `nn.ReLU`, `nn.LeakyReLU` 等模块替换自定义函数
- 添加更多激活函数如 GELU、Swish 等现代变体
