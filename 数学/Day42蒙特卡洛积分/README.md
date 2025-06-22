使用 **随机采样（Monte Carlo 方法）** 来估计复杂积分是一种非常强大的数值方法，尤其适用于高维空间或解析解难以获得的积分问题。

---

## 🧮 一、基本思想：蒙特卡洛积分

考虑一个定积分：

$$
I = \int_a^b f(x) dx
$$

我们可以用 **均匀随机采样** 的方式近似这个积分：

1. 在区间 $[a, b]$ 上生成 $N$ 个独立同分布的随机数 $x_1, x_2, ..., x_N$
2. 计算函数值的平均：
   $$
   I \approx (b - a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)
   $$

> ✅ 这就是 **蒙特卡洛积分的基本形式**

---

## 🧪 二、Python 示例：用随机采样估计积分

我们来估计下面这个复杂积分：

$$
I = \int_0^1 e^{-x^2} dx
$$

这个积分没有初等函数的解析解，但可以通过 Monte Carlo 方法进行估计。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义被积函数
def integrand(x):
    return np.exp(-x**2)

# 积分区间
a, b = 0, 1

# 随机采样点数量
N = 1000000

# Step 1: 随机采样
x_samples = np.random.uniform(a, b, N)

# Step 2: 计算函数值并求平均
f_values = integrand(x_samples)
integral_estimate = (b - a) * np.mean(f_values)

print("积分估计值:", integral_estimate)
```

### 🔍 输出示例：

```
积分估计值: 0.746824
```

> 真实值约为：`scipy.special.erf(1) / 2 ≈ 0.746824`

---

## 📊 三、可视化 Monte Carlo 估计过程（可选）

如果你想观察随着样本数量增加，估计值如何收敛，可以添加如下代码：

```python
# 计算前缀平均
cumulative_mean = np.cumsum(f_values) / np.arange(1, N+1)
integral_convergence = (b - a) * cumulative_mean

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(integral_convergence[:10000], label='Monte Carlo Estimate')
plt.axhline(y=0.746824, color='r', linestyle='--', label='True Value')
plt.xlabel("Number of Samples")
plt.ylabel("Integral Estimate")
plt.title("Monte Carlo Integration Convergence")
plt.grid()
plt.legend()
plt.show()
```

---

## 🧠 四、扩展：多维积分估计

对于多维积分，例如：

$$
I = \int_0^1 \int_0^1 e^{-(x^2 + y^2)} dx dy
$$

只需要将采样扩展到二维即可：

```python
def integrand_2d(x, y):
    return np.exp(-(x**2 + y**2))

# 二维积分区域 [0,1] × [0,1]
N = 100000
x_samples = np.random.uniform(0, 1, N)
y_samples = np.random.uniform(0, 1, N)

f_values = integrand_2d(x_samples, y_samples)
integral_estimate = (1 - 0) * (1 - 0) * np.mean(f_values)

print("二维积分估计值:", integral_estimate)
```

---

## ✅ 五、总结对比表

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 蒙特卡洛积分 | 易于实现，适合高维 | 收敛慢（O(1/√N)） | 高维积分、复杂函数、概率积分 |
| 梯形法 / Simpson 法 | 收敛快 | 难以处理高维 | 低维光滑函数积分 |

---

## 📌 六、应用场景举例

| 应用领域 | 示例 |
|----------|------|
| 金融工程 | 期权定价（Black-Scholes 模型） |
| 物理模拟 | 多粒子系统能量积分 |
| 机器学习 | 贝叶斯推断中的后验期望计算 |
| 图形学 | 渲染方程中的光照积分计算 |

---