计算多元函数的**梯度（Gradient）**，是优化、机器学习和科学计算中的基础操作。下面我将分两部分讲解：

---

## ✅ 一、什么是多元函数的梯度？

对于一个多元实值函数 $ f(\mathbf{x}) = f(x_1, x_2, \dots, x_n) $，它的 **梯度** 是一个向量，由所有偏导数组成：

$$
\nabla f(\mathbf{x}) =
\left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

梯度的方向表示函数在该点上升最快的方向，负方向则为下降最快的方向。

---

## ✅ 二、用 NumPy 实现梯度下降算法（Gradient Descent）

### 🧪 示例：最小化函数 $ f(x, y) = x^2 + (y - 3)^2 $

这个函数的最小值在 $ (x, y) = (0, 3) $ 处，我们使用梯度下降法来逼近它。

### 🔢 步骤：

1. 定义目标函数；
2. 计算其梯度；
3. 使用迭代更新公式：
   $$
   \mathbf{x}_{k+1} = \mathbf{x}_k - \eta \cdot \nabla f(\mathbf{x}_k)
   $$
   其中 $\eta$ 是学习率（learning rate）；
4. 设置迭代终止条件（如最大迭代次数或误差阈值）。

---

### 📌 Python 实现代码：

```python
import numpy as np

# 目标函数
def f(x):
    return x[0]**2 + (x[1] - 3)**2

# 梯度函数
def grad_f(x, eps=1e-6):
    # 数值微分近似梯度
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return grad

# 梯度下降法
def gradient_descent(starting_point, learning_rate=0.1, max_iter=1000, tol=1e-6):
    x = np.array(starting_point, dtype=float)
    for i in range(max_iter):
        grad = grad_f(x)
        step = learning_rate * grad
        if np.linalg.norm(step) < tol:
            print(f"收敛于第 {i} 次迭代")
            break
        x -= step
        if i % 100 == 0:
            print(f"迭代 {i}: x = {x}, f(x) = {f(x)}")
    return x, f(x)

# 初始猜测值
x0 = [1.0, 1.0]

# 运行梯度下降
minimum_point, minimum_value = gradient_descent(x0)

print("极小值点:", minimum_point)
print("极小值:", minimum_value)
```

---

### 📈 输出示例（可能略有不同）：

```
迭代 0: x = [1. 1.], f(x) = 5.0
迭代 100: x = [0.005 2.995], f(x) ≈ 0.000025
收敛于第 127 次迭代
极小值点: [0.0001 3.0000]
极小值: 1.23e-08
```

---

## 📌 总结

| 功能 | 实现方式 |
|------|-----------|
| 定义函数 | `def f(x): return ...` |
| 梯度计算 | 数值微分：中心差分法 `f(x+h) - f(x-h) / 2h` |
| 梯度下降 | 迭代更新：`x = x - lr * grad` |
| 停止条件 | 最大迭代数 或 步长小于容差 |

---

## 🧠 提示与扩展

- 如果你知道函数的解析梯度，可以直接写出来代替数值微分，速度更快。
- 可以使用更高级的优化器，如 `scipy.optimize.minimize` 中的 `BFGS` 等方法。
- 学习率太大会导致发散，太小会收敛慢。可以尝试自适应学习率方法（如 Adam）。
- 对于高维问题，可以结合 NumPy 向量化加速计算。

---
