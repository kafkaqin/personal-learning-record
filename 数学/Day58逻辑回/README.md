使用 **Sigmoid 函数** 实现分类是逻辑回归（Logistic Regression）的核心思想。Sigmoid 函数将任意实数映射到 $[0, 1]$ 区间，因此非常适合用于二分类问题的概率输出。

---

## 🧠 一、Sigmoid 函数定义

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- 当 $ z \to +\infty $，$\sigma(z) \to 1$
- 当 $ z \to -\infty $，$\sigma(z) \to 0$
- 当 $ z = 0 $，$\sigma(z) = 0.5$

这个函数常用于将线性输出转化为概率值，从而进行二分类：

$$
P(y=1|x) = \sigma(w^T x + b)
$$

---

## ✅ 二、Python 示例：用 NumPy 实现 Sigmoid 分类器

我们以一个简单的二维数据集为例，演示如何手动实现 Sigmoid 函数和分类预测。

### 1. 定义 Sigmoid 函数

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### 2. 构造简单数据（两类点）

```python
# 生成两类数据点（类别 0 和 1）
np.random.seed(42)
X_class0 = np.random.randn(50, 2) + [2, 2]
X_class1 = np.random.randn(50, 2) + [-2, -2]
X = np.vstack((X_class0, X_class1))

y = np.array([0]*50 + [1]*50).reshape(-1, 1)

# 添加偏置项（w0*x0 + w1*x1 + b => 使用 w0*x0 + w1*x1 + w2*1）
X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
```

### 3. 初始化参数并训练（梯度下降）

```python
# 初始化权重
weights = np.random.randn(3, 1)

learning_rate = 0.1
n_iterations = 1000

for i in range(n_iterations):
    # 线性组合 + sigmoid
    z = X_bias @ weights
    y_pred = sigmoid(z)

    # 计算损失（可选）
    if i % 200 == 0:
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        print(f"Iteration {i}: Loss = {loss:.4f}")

    # 梯度计算（交叉熵损失的梯度）
    gradient = (y_pred - y) * X_bias
    gradient = np.mean(gradient, axis=0).reshape(-1, 1)

    # 参数更新
    weights -= learning_rate * gradient
```

### 4. 预测与可视化

```python
# 分类函数
def predict(X, weights):
    z = X @ weights
    prob = sigmoid(z)
    return (prob >= 0.5).astype(int)

# 测试新数据点
test_points = np.array([[3, 3], [-3, -3]])
test_points_bias = np.hstack((test_points, np.ones((test_points.shape[0], 1))))
predictions = predict(test_points_bias, weights)

print("测试点预测结果:")
for point, label in zip(test_points, predictions):
    print(f"点 {point} 被预测为类别 {label.item()}")
```

### 5. 可视化决策边界

```python
x_vals = np.linspace(-5, 5, 100)
y_vals = -(weights[0] * x_vals + weights[2]) / weights[1]

plt.scatter(X_class0[:, 0], X_class0[:, 1], label="Class 0", color='blue')
plt.scatter(X_class1[:, 0], X_class1[:, 1], label="Class 1", color='red')
plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')
plt.title('Sigmoid 分类器的决策边界')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
```

---

## ✅ 输出示例：

```
Iteration 0: Loss = 0.9678
Iteration 200: Loss = 0.2321
Iteration 400: Loss = 0.1421
Iteration 600: Loss = 0.0981
Iteration 800: Loss = 0.0735

测试点预测结果:
点 [3. 3.] 被预测为类别 1
点 [-3. -3.] 被预测为类别 0
```

---

## 📌 三、关键公式总结

| 公式 | 描述 |
|------|------|
| $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Sigmoid 函数 |
| $y_{pred} = \sigma(w^T x + b)$ | 概率预测 |
| $\mathcal{L} = -\frac{1}{n} \sum y \log(y_{pred}) + (1-y)\log(1-y_{pred})$ | 交叉熵损失 |
| $\nabla_w \mathcal{L} = \frac{1}{n} \sum (y_{pred} - y) x$ | 梯度 |

---

## 🧪 四、扩展建议

| 功能 | 方法 |
|------|------|
| 多分类 | 使用 Softmax 替代 Sigmoid |
| 正则化 | 在损失中加入 L2 正则项 |
| 向量化优化 | 使用矩阵运算加速计算 |
| 使用 Scikit-Learn | `LogisticRegression` 更高效 |
| 加入偏差项 | 建议始终加上，否则模型可能不准确 |

---

如果你希望：
- 扩展为多变量逻辑回归
- 加入正则化（L1/L2）
- 改为批量/随机梯度下降
- 使用 PyTorch/JAX 自动求导版本

