使用 **有限差分法（Finite Difference Method, FDM）** 解 **热传导方程（Heat Equation）** 是数值求解偏微分方程中最经典的方法之一。

---

## 🌡️ 一、热传导方程简介

我们考虑最简单的 **一维热传导方程（扩散方程）**：

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \; t \in [0, T]
$$

其中：

- $ u(x,t) $：温度分布（未知函数）
- $ \alpha $：热扩散率（常数）
- 初始条件：$ u(x, 0) = f(x) $
- 边界条件：例如 Dirichlet 条件 $ u(0,t) = a,\; u(L,t) = b $

---

## 🔢 二、有限差分法基本思想

将时间和空间离散化，构造网格：

- 时间步长：$ \Delta t $
- 空间步长：$ \Delta x $
- 在每个网格点上用差商近似导数

### ⚙️ 显式格式（向前时间、中心空间）

对时间导数使用向前差分，对空间导数使用中心差分：

$$
\frac{u_i^{n+1} - u_i^n}{\Delta t} = \alpha \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{(\Delta x)^2}
$$

整理得更新公式：

$$
u_i^{n+1} = u_i^n + \nu (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
$$

其中：
$$
\nu = \alpha \frac{\Delta t}{(\Delta x)^2}
$$

> ✅ **稳定性条件（CFL 条件）**：必须满足 $ \nu \leq 0.5 $，否则结果会发散。

---

## 🧪 三、Python 实现代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 1.0             # 空间长度
T = 0.5             # 总时间
alpha = 0.01        # 热扩散率
Nx = 50             # 空间网格数
Nt = 1000           # 时间步数
dx = L / (Nx - 1)
dt = T / Nt
nu = alpha * dt / dx**2

print(f"ν = {nu:.4f}, 稳定性要求 ν ≤ 0.5")

# 初始化网格
x = np.linspace(0, L, Nx)
u = np.zeros((Nx, Nt+1))

# 初始条件：三角形分布
u[:, 0] = np.sin(np.pi * x)

# 边界条件
u[0, :] = 0
u[-1, :] = 0

# 显式迭代
for n in range(Nt):
    for i in range(1, Nx-1):
        u[i, n+1] = u[i, n] + nu * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

# 可视化
plt.figure(figsize=(10, 6))
for n in [0, 10, 50, 100, 200, 500, 1000]:
    plt.plot(x, u[:, n], label=f't={n*dt:.3f}s')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("热传导方程的有限差分解")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📈 四、输出说明

- 每条曲线表示在某个时刻 $ t $ 的温度分布 $ u(x,t) $
- 随着时间推移，初始的正弦波逐渐“扩散”并趋于平稳
- 最终趋向于稳定状态：边界为 0，中间也趋近于 0

---

## 📌 五、扩展建议

| 扩展方向 | 方法 |
|----------|------|
| 更高精度 | 使用 Crank-Nicolson 格式（隐式，无条件稳定） |
| 多维问题 | 推广到二维热传导方程（需要二维数组和更多邻域计算） |
| 非均匀材料 | 让 $ \alpha = \alpha(x) $，变成变系数 PDE |
| 非线性问题 | 如含源项或非线性扩散项 |
| 并行计算 | 使用 NumPy 向量化代替双重循环加速 |

---

## 🧠 六、应用场景举例

| 应用领域 | 示例 |
|----------|------|
| 物理 | 温度场演化、分子扩散 |
| 工程 | 半导体热管理、建筑热传导分析 |
| 数学金融 | Black-Scholes 方程（期权定价） |
| 图像处理 | 图像平滑、各向异性扩散滤波器 |

---