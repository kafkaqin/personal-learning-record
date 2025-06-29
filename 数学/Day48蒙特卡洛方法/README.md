使用 **蒙特卡洛方法（Monte Carlo Simulation）** 来计算圆周率 π 是一个经典的教学示例，它展示了如何通过随机采样来近似复杂问题的解。

我们也可以扩展这个思想来模拟路径积分（如粒子在随机路径中的期望值），但本回答将从最基础的 **π 值估计** 开始，并逐步扩展到更复杂的 **路径积分模拟**。

---

## 🧮 一、用蒙特卡洛方法估算 π 的值

### 📌 思路：

- 在边长为 2 的正方形内画一个单位圆（半径 = 1）
- 随机生成点 $(x, y)$，落在该正方形中
- 如果 $x^2 + y^2 \leq 1$，则该点在圆内
- 统计落在圆内的点的比例，乘以正方形面积即可估计圆面积，从而得到 π

$$
\frac{\text{圆内点数}}{\text{总点数}} \approx \frac{\pi r^2}{(2r)^2} = \frac{\pi}{4}
\Rightarrow \pi \approx 4 \cdot \frac{\text{圆内点数}}{\text{总点数}}
$$

---

### ✅ Python 实现代码：

```python
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
num_samples = 100000

# 生成随机点 (x, y) ∈ [-1, 1] × [-1, 1]
x = np.random.uniform(-1, 1, num_samples)
y = np.random.uniform(-1, 1, num_samples)

# 判断是否在圆内
in_circle = x**2 + y**2 <= 1
count_in_circle = np.sum(in_circle)

# 估计 π
pi_estimate = 4 * count_in_circle / num_samples

print(f"估计 π 值: {pi_estimate:.6f}")

# 可视化部分点（只显示前1000个）
plt.figure(figsize=(6, 6))
plt.scatter(x[:1000], y[:1000], c=in_circle[:1000], cmap='coolwarm', s=5)
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', lw=2)
plt.title(f"蒙特卡洛估计 π ≈ {pi_estimate:.6f}")
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

---

### ✅ 输出示例：

```
估计 π 值: 3.14396
```

> 随着 `num_samples` 增大，结果会越来越接近真实 π 值（≈3.14159）

---

## 🧪 二、扩展：路径积分模拟（Path Integral）

我们可以将蒙特卡洛思想扩展到 **路径积分（Path Integral）**，即对所有可能路径的加权求和。这在量子力学、金融期权定价等领域有广泛应用。

### 示例：布朗运动下的路径积分

我们模拟一个粒子从原点出发，在一段时间内做一维布朗运动，并估计其位移平方的平均值。

#### 数学表示：

$$
\langle x^2(t) \rangle = \mathbb{E}[x^2(t)] \approx \frac{1}{N}\sum_{i=1}^{N} x_i^2(t)
$$

其中每个 $x_i(t)$ 是一条布朗路径。

---

### ✅ Python 实现代码：

```python
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 1.0           # 时间长度
dt = 0.01         # 时间步长
steps = int(T / dt)
num_paths = 1000  # 模拟路径数量

# 初始化路径数组
paths = np.zeros((num_paths, steps))

# 生成布朗路径
for i in range(num_paths):
    dW = np.random.normal(0, np.sqrt(dt), steps)
    paths[i, :] = np.cumsum(dW)

# 计算最终位置的均方值
mean_square_displacement = np.mean(paths[:, -1]**2)

print(f"⟨x²⟩ ≈ {mean_square_displacement:.4f}")

# 可视化几条路径
plt.figure(figsize=(10, 5))
t = np.arange(0, T, dt)
for i in range(10):  # 显示前10条路径
    plt.plot(t, paths[i, :], alpha=0.7)
plt.title("布朗运动路径模拟")
plt.xlabel("时间 t")
plt.ylabel("位置 x(t)")
plt.grid(True)
plt.show()
```

---

### ✅ 输出示例：

```
⟨x²⟩ ≈ 0.9832
```

> 理论上 $\langle x^2(t) \rangle = t = 1$，所以数值与理论值非常接近。

---

## 📌 三、总结对比表

| 方法 | 应用场景 | 特点 |
|------|----------|------|
| 蒙特卡洛估计 π | 教学演示、简单积分 | 直观、适合并行计算 |
| 路径积分模拟 | 布朗运动、期权定价、量子力学 | 可模拟复杂系统行为 |

---

## 🧠 四、进阶建议

| 扩展方向 | 方法 |
|----------|------|
| 提高精度 | 使用重要性采样（Importance Sampling） |
| 并行加速 | 使用 NumPy 向量化或多线程/GPU |
| 更复杂模型 | 加入漂移项、扩散系数、边界条件等 |
| 金融应用 | 用于 Black-Scholes 模型中的期权定价 |

---
