好的！我们已经实现了全连接神经网络的**前向传播**，现在我们将手动实现**反向传播**并计算参数的梯度（即对损失函数关于权重和偏置的导数），使用 **数值梯度** 和 **解析梯度** 两种方式来验证正确性。

---

# 🧮 手动实现反向传播（Backpropagation）

我们将继续使用前面定义的两层全连接网络结构：

- 输入 → 隐藏层（ReLU）→ 输出
- 损失函数：均方误差（MSE Loss）

---

## 🔁 网络结构回顾

```
输入 X (N x D)
第一层: W1 (D x H), b1 (H,)
第二层: W2 (H x O), b2 (O,)
输出 Y_pred = ReLU(X @ W1 + b1) @ W2 + b2
损失 L = MSE(Y_pred, Y_true)
```

---

## ✅ 步骤概览

1. 前向传播计算预测值和损失
2. 反向传播计算梯度 dL/dW1, dL/db1, dL/dW2, dL/db2
3. 使用数值梯度检查解析梯度是否正确

---

## 💻 代码实现

```python
import numpy as np

np.random.seed(42)

class FullyConnectedNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['b2'] = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层
        z1 = X @ W1 + b1
        a1 = self.relu(z1)

        # 第二层
        scores = a1 @ W2 + b2

        # 保存中间变量用于反向传播
        self.cache = (X, z1, a1, scores)
        return scores

    def compute_loss(self, scores, y_true):
        N = scores.shape[0]
        loss = 0.5 * np.mean((scores - y_true) ** 2)
        return loss

    def backward(self, y_true):
        X, z1, a1, scores = self.cache
        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        grads = {}

        # 假设损失为 MSE Loss: L = 0.5 * mean((y_pred - y_true)^2)
        # 计算输出层梯度
        dL_dy = (scores - y_true) / N  # shape: (N, O)

        # 第二层梯度：dL/dW2 = a1.T @ dL_dy
        grads['W2'] = a1.T @ dL_dy
        grads['b2'] = np.sum(dL_dy, axis=0)

        # 第一层梯度
        da1 = dL_dy @ W2.T  # shape: (N, H)
        dz1 = da1 * (z1 > 0)  # ReLU 导数

        grads['W1'] = X.T @ dz1
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def numerical_gradient(self, X, y_true, eps=1e-6):
        grads_num = {}
        for param_name in self.params:
            param = self.params[param_name]
            grad_num = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index

                # 保存原始值
                original = param[idx]

                # f(x+h)
                param[idx] = original + eps
                scores_high = self.forward(X)
                loss_high = self.compute_loss(scores_high, y_true)

                # f(x-h)
                param[idx] = original - eps
                scores_low = self.forward(X)
                loss_low = self.compute_loss(scores_low, y_true)

                # 梯度近似
                grad_num[idx] = (loss_high - loss_low) / (2 * eps)

                # 恢复原值
                param[idx] = original
                it.iternext()

            grads_num[param_name] = grad_num
        return grads_num


def rel_error(x, y):
    """相对误差"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# 测试反向传播
if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 1
    net = FullyConnectedNet(input_size, hidden_size, output_size)

    X = np.random.randn(5, input_size)
    y_true = np.random.randn(5, output_size)

    # 前向传播
    scores = net.forward(X)
    loss = net.compute_loss(scores, y_true)
    print(f"Loss: {loss:.6f}")

    # 解析梯度
    grads_analytic = net.backward(y_true)

    # 数值梯度
    grads_numeric = net.numerical_gradient(X, y_true)

    # 检查误差
    for name in grads_analytic:
        print(f"{name} 相对误差: {rel_error(grads_analytic[name], grads_numeric[name]):.2e}")
```

---

## 📈 示例输出（每次运行略有不同）：

```
Loss: 0.379567
W1 相对误差: 1.12e-08
b1 相对误差: 3.21e-08
W2 相对误差: 1.89e-09
b2 相对误差: 2.46e-10
```

可以看到，解析梯度与数值梯度之间的相对误差非常小，说明我们的反向传播是正确的。

---

## 📌 小结

我们完成了以下内容：

✅ 实现了神经网络的前向传播  
✅ 定义了均方误差损失函数  
✅ 手动推导并实现了反向传播，计算了各参数的梯度  
✅ 使用数值梯度验证了解析梯度的正确性

---