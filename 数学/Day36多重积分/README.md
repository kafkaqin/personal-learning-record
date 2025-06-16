当然可以！在 Python 中，使用 **`scipy.integrate.dblquad`** 可以非常方便地计算一个函数的 **二重积分（double integral）**。

---

## ✅ 一、基本用法：`dblquad(func, a, b, gfun, hfun)`

### 📌 函数签名：

```python
scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8)
```

| 参数 | 含义 |
|------|------|
| `func(x, y)` | 被积函数（先对 y 积分，再对 x 积分） |
| `a`, `b` | 外层积分变量 $ x $ 的上下限 |
| `gfun(x)` | 内层积分下限（关于 x 的函数） |
| `hfun(x)` | 内层积分上限（关于 x 的函数） |

---

## ✅ 二、示例：计算单位正方形上的二重积分

我们来计算下面这个积分：

$$
\int_{0}^{1} \int_{0}^{1} (x^2 + y^2) \, dy dx
$$

### 🧪 Python 示例代码：

```python
from scipy.integrate import dblquad
import numpy as np

# 定义被积函数 f(x, y)
def integrand(y, x):  # 注意参数顺序：y 在前，x 在后！
    return x**2 + y**2

# 计算积分 ∫(x=0到1) ∫(y=0到1) (x² + y²) dy dx
result, error = dblquad(
    func=integrand,
    a=0, b=1,          # x 的积分范围 [0, 1]
    gfun=lambda x: 0,  # y 的下限
    hfun=lambda x: 1   # y 的上限
)

print("积分结果:", result)
print("误差估计:", error)
```

### 🔍 输出结果：

```
积分结果: 0.6666666666666666
误差估计: 7.401486830834377e-15
```

> 即：$ \frac{2}{3} $

---

## ✅ 三、更复杂的例子：积分区域是圆盘

我们来计算如下积分：

$$
\int\int_{x^2 + y^2 \leq 1} (x^2 + y^2) \, dx dy
$$

我们可以转换为极坐标，但也可以直接使用 `dblquad` 来做笛卡尔坐标下的积分。

### 🧪 实现代码：

```python
def integrand(y, x):
    return x**2 + y**2

# 积分区域：x ∈ [-1, 1], y ∈ [-sqrt(1 - x²), sqrt(1 - x²)]
result, error = dblquad(
    func=integrand,
    a=-1,
    b=1,
    gfun=lambda x: -np.sqrt(1 - x**2),
    hfun=lambda x: np.sqrt(1 - x**2)
)

print("积分结果:", result)
print("误差估计:", error)
```

### 🔍 输出结果：

```
积分结果: 1.5707963267948966
误差估计: 1.7439342498925556e-14
```

> 这个值等于 $ \frac{\pi}{2} $，因为：
> $$
> \iint_{x^2 + y^2 \leq 1} (x^2 + y^2) dxdy = \int_0^{2\pi} \int_0^1 r^2 \cdot r dr d\theta = \frac{\pi}{2}
> $$

---

## ✅ 四、注意事项

- `dblquad` 的参数顺序是 `func(y, x)`，即 **内层变量在前**，外层变量在后。
- 如果你有额外参数要传给被积函数，可以使用 `args=(...)`。
- 返回两个值：`result` 是积分值，`error` 是误差估计。

---

## ✅ 五、应用场景举例

| 场景 | 应用方式 |
|------|----------|
| 物理模拟 | 计算质量、电荷分布等 |
| 概率统计 | 计算联合概率密度的积分 |
| 数学建模 | 求解面积/体积、期望等 |
| 工程分析 | 有限元方法中的局部积分 |

---
