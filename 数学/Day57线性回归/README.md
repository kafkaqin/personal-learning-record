使用 **NumPy** 实现线性回归是一个很好的练习，可以帮助你理解线性回归的数学原理和梯度下降优化过程。我们将手动实现：

- 线性模型：$ y = wx + b $
- 损失函数：均方误差（MSE）
- 梯度下降更新规则
- 参数迭代训练

---

## 📌 一、问题定义

我们要拟合一个线性模型：

$$
y = wx + b
$$

目标是通过最小化均方误差（MSE）来找到最优参数 $ w $ 和 $ b $。

### 均方误差公式：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - (wx_i + b))^2
$$

---

## ✅ 二、Python 实现代码（使用 NumPy）

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + 噪声

# 2. 初始化参数
w = 0.0
b = 0.0

learning_rate = 0.1
n_iterations = 1000
m = len(X)

# 3. 梯度下降循环
for iteration in range(n_iterations):
    y_pred = w * X + b
    error = y_pred - y

    # 计算梯度
    gradient_w = (2 / m) * np.dot(X.T, error).item()
    gradient_b = (2 / m) * np.sum(error)

    # 更新参数
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # 每隔 100 次记录损失
    if iteration % 100 == 0:
        loss = np.mean(error**2)
        print(f"Iteration {iteration}: Loss = {loss:.4f}")

# 4. 输出最终结果
print("\n最终模型参数:")
print(f"斜率 w = {w:.4f}, 截距 b = {b:.4f}")

# 5. 可视化结果
plt.scatter(X, y, color='blue', label='数据点')
plt.plot(X, w * X + b, color='red', label='拟合直线')
plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归拟合")
plt.legend()
plt.grid(True)
plt.show()
```

---

## ✅ 输出示例：

```
Iteration 0: Loss = 78.6967
Iteration 100: Loss = 0.7226
Iteration 200: Loss = 0.6252
...
Iteration 900: Loss = 0.5983

最终模型参数:
斜率 w = 3.0012, 截距 b = 3.9983
```

---

## 📈 三、可视化输出说明

- 散点图表示原始数据（带噪声）
- 红色直线是我们通过梯度下降学习到的最佳拟合直线
- 最终参数接近真实值 `w=3`, `b=4`，说明模型学习效果良好

---

## 🧠 四、关键步骤解释

| 步骤 | 描述 |
|------|------|
| 数据准备 | 使用 NumPy 生成模拟数据 |
| 初始化参数 | 设置初始的斜率和截距 |
| 预测与误差 | 计算当前预测值和误差 |
| 梯度计算 | 根据 MSE 求导公式计算梯度 |
| 参数更新 | 使用学习率进行梯度下降更新 |
| 收敛监控 | 打印每轮迭代的损失值 |

---

## 📌 五、扩展建议

| 功能 | 方法 |
|------|------|
| 向量化加速 | 使用 NumPy 的广播机制和向量运算 |
| 多变量支持 | 将 `X` 扩展为二维矩阵，使用 `X.dot(w)` |
| 添加偏置项 | 使用 `np.hstack([np.ones((m,1)), X])` 构建设计矩阵 |
| 学习率调整 | 使用自适应学习率算法（如 AdaGrad、Adam） |
| 使用损失函数类 | 自定义封装 `compute_loss()` 函数 |

---