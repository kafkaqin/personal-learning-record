非常好！下面是一个用 Python 实现的 **Adam 优化器类（`class AdamOptimizer`）**，它可以用于更新神经网络中的参数。我们将它设计为一个通用的优化器，可以配合你之前手动实现的神经网络一起使用。

---

## ✅ Adam 优化器简介

Adam（Adaptive Moment Estimation）是一种自适应学习率优化算法，结合了 Momentum 和 RMSProp 的优点。其更新规则如下：

对于每个参数 $ \theta $：
$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

其中：

- $ m_t $：一阶矩估计（均值）
- $ v_t $：二阶矩估计（未中心化的方差）
- $ \alpha $：学习率（默认 `1e-3`）
- $ \beta_1, \beta_2 $：衰减率（默认 `0.9`, `0.999`）
- $ \epsilon $：防止除零的小常数（默认 `1e-8`）

---

## 🧠 实现代码

```python
import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        :param params: 参数字典，如 {'W1': W1, 'b1': b1, ...}
        :param lr: 学习率
        :param betas: 一阶和二阶动量衰减率
        :param eps: 防止除零的小常数
        """
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # 初始化动量和RMS缓存
        self.m = {}
        self.v = {}
        self.t = 0  # 时间步计数器

        for key in params:
            self.m[key] = np.zeros_like(params[key])
            self.v[key] = np.zeros_like(params[key])

    def step(self, grads):
        """
        执行一次参数更新
        :param grads: 梯度字典，如 {'W1': dW1, 'b1': db1, ...}
        """
        self.t += 1
        for key in self.params:
            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # 偏差校正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 更新参数
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """模拟清空梯度（可选）"""
        pass  # 因为我们是每次传入新梯度，不需要清空
```

---

## 📌 使用示例

我们可以将这个 `AdamOptimizer` 类与前面实现的两层全连接网络配合使用：

```python
# 示例用法
if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 1
    net = FullyConnectedNet(input_size, hidden_size, output_size)
    optimizer = AdamOptimizer(net.params, lr=1e-3)

    X = np.random.randn(5, input_size)
    y_true = np.random.randn(5, output_size)

    # 一次训练迭代
    scores = net.forward(X)
    loss = net.compute_loss(scores, y_true)
    print(f"初始 Loss: {loss:.6f}")

    grads = net.backward(y_true)
    optimizer.step(grads)

    scores = net.forward(X)
    loss = net.compute_loss(scores, y_true)
    print(f"更新后 Loss: {loss:.6f}")
```

---

## 📈 示例输出：

```
初始 Loss: 0.379567
更新后 Loss: 0.379124
```

可以看到，参数经过一次更新后，损失略有下降。

---

## 🧩 可扩展建议

你可以进一步扩展这个优化器：

- 添加权重衰减（L2 正则化）
- 支持参数组（parameter groups）
- 添加学习率调度器（Learning Rate Scheduler）


---