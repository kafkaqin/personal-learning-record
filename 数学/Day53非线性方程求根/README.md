使用 **牛顿迭代法（Newton-Raphson Method）** 是一种求解非线性方程 $ f(x) = 0 $ 的经典数值方法。它利用函数的导数信息，通过迭代快速逼近根。

在 Python 中，`scipy.optimize.newton` 提供了这一功能，支持多种实现方式（如标准牛顿法、割线法等），适用于光滑且可导的函数。

---

## 🧮 一、牛顿法基本原理

给定一个连续可导函数 $ f(x) $，其根（即 $ f(x) = 0 $ 的解）可以通过以下迭代公式逐步逼近：

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

初始值 $ x_0 $ 需要选择得足够接近真实根以确保收敛。

---

## ✅ 二、Python 示例：使用 `scipy.optimize.newton`

### 示例目标：
我们来求解方程：

$$
f(x) = x^3 - 2x - 5
$$

这个方程在 $ x \approx 2.09455 $ 处有一个实根。

```python
from scipy.optimize import newton
import numpy as np

# 定义函数
def f(x):
    return x**3 - 2*x - 5

# 定义导数（如果提供，newton 方法会使用牛顿法；否则使用割线法）
def df(x):
    return 3*x**2 - 2

# 初始猜测值
x0 = 2.0

# 使用牛顿法求解
root = newton(f, x0=x0, fprime=df, tol=1e-8, maxiter=50)

print("找到的根 x =", root)
print("验证 f(x) ≈", f(root))
```

---

## ✅ 输出示例：

```
找到的根 x = 2.0945514815423265
验证 f(x) ≈ 8.881784197001252e-16
```

误差非常小，说明已成功找到近似解。

---

## 📌 三、参数说明（`newton` 函数）

| 参数 | 含义 |
|------|------|
| `func` | 要求解的函数 $ f(x) $ |
| `x0` | 初始猜测值 |
| `fprime` | 可选，函数的导数 $ f'(x) $ |
| `tol` | 收敛容忍度，默认为 `1.48e-8` |
| `maxiter` | 最大迭代次数 |
| `full_output` | 若为 True，返回额外信息（如迭代次数） |

---

## 🧪 四、不提供导数时：割线法（Secant Method）

如果不提供导数，`newton()` 将自动使用 **割线法（Secant Method）**，只需两个初始点：

```python
# 不需要导数 df
root_secant = newton(f, x0=2.0, tol=1e-8, maxiter=50)
```

虽然收敛速度略慢于牛顿法（超线性 vs 二次），但避免了计算导数。

---

## 📈 五、可视化牛顿法过程（可选）

你可以绘制函数图像和每次迭代的位置，帮助理解算法行为：

```python
import matplotlib.pyplot as plt

x = np.linspace(1.5, 3.0, 400)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = x^3 - 2x - 5$')
plt.axhline(0, color='black', linewidth=0.5)

# 显示根位置
plt.plot(root, f(root), 'ro', label='Root')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('牛顿法求根示意图')
plt.grid(True)
plt.legend()
plt.show()
```

---

## 🧠 六、注意事项与常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 不收敛 | 初值离根太远或函数不可导 | 尝试换初值或检查函数是否平滑 |
| 导数为零 | 牛顿法失效 | 检查导数表达式，尝试使用割线法 |
| 多个根 | 找到的是局部最近的根 | 可尝试多个初值进行搜索 |

---

## 📌 七、应用场景举例

| 应用领域 | 示例 |
|----------|------|
| 工程计算 | 求解电路中的非线性方程 |
| 经济学 | 求市场均衡价格 |
| 数值分析 | 解非线性微分方程的隐式格式 |
| 金融工程 | 计算期权波动率（Implied Volatility） |

---

