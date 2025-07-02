使用 `scipy.optimize.linprog` 是 Python 中解决 **线性规划（Linear Programming）** 问题的标准方法之一。它可以用于最小化一个线性目标函数，同时满足一系列线性等式或不等式约束。

---

## 📌 一、线性规划标准形式（linprog 要求）

`scipy.optimize.linprog` 默认接受以下标准形式：

$$
\begin{aligned}
\min \quad & c^T x \\
\text{s.t.} \quad & A_{ub} \cdot x \leq b_{ub} \\
& A_{eq} \cdot x = b_{eq} \\
& l \leq x \leq u
\end{aligned}
$$

其中：
- `c`: 目标函数系数向量
- `A_ub`, `b_ub`: 不等式约束矩阵和向量
- `A_eq`, `b_eq`: 等式约束矩阵和向量
- `bounds`: 每个变量的上下限（默认为 $x_i \geq 0$）

---

## 🧪 二、示例问题

我们来解一个经典的线性规划问题：

### 🎯 目标函数：
$$
\min \quad -3x_1 - 5x_2
$$

### 📚 约束条件：
$$
\begin{aligned}
3x_1 + 2x_2 &\leq 18 \\
x_1 &\leq 4 \\
x_2 &\leq 6 \\
x_1, x_2 &\geq 0
\end{aligned}
$$

---

## ✅ 三、Python 实现代码

```python
from scipy.optimize import linprog

# 定义目标函数系数（注意：linprog 是最小化，所以最大化要取负）
c = [-3, -5]  # 最小化 -3x1 -5x2 等价于最大化 3x1 +5x2

# 不等式约束 Ax <= b
A_ub = [
    [3, 2],   # 3x1 + 2x2 <= 18
    [1, 0],   # x1 <= 4
    [0, 1]    # x2 <= 6
]
b_ub = [18, 4, 6]

# 变量范围 (x1 >= 0, x2 >= 0)
bounds = [(0, None), (0, None)]  # None 表示无上限

# 求解
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# 输出结果
print("是否找到最优解:", result.success)
print("最优值:", -result.fun)  # 因为我们是求最大化的转换
print("最优解 x1, x2:", result.x)
```

---

## ✅ 四、输出结果示例：

```
是否找到最优解: True
最优值: 36.0
最优解 x1, x2: [2. 6.]
```

解释：
- 在 $x_1 = 2, x_2 = 6$ 处达到最大值 36。
- 这与手工计算一致，符合约束条件。

---

## 📈 五、可视化验证（可选）

对于二维问题，可以画出可行域和目标函数方向，验证最优解位置。

```python
import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0, 5, 400)
x2 = np.linspace(0, 7, 400)
X1, X2 = np.meshgrid(x1, x2)

# 绘制约束区域
plt.figure(figsize=(8,6))
plt.fill_betweenx(x2, 0, (18 - 2*x2)/3, where=(x2 <= 6), color='red', alpha=0.1, label='3x1+2x2 ≤ 18')
plt.fill_between(x1, 0, 6, where=(x1 <=4), color='blue', alpha=0.1, label='x2 ≤ 6 and x1 ≤4')

# 目标函数等高线
Z = 3*X1 + 5*X2
CS = plt.contour(X1, X2, Z, colors='green', linestyles='dashed')
plt.clabel(CS, inline=True, fontsize=8)

# 最优解点
plt.plot(2, 6, 'ro', label='Optimal Solution (2,6)')

plt.xlim(0, 5)
plt.ylim(0, 7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('线性规划可行域与最优解')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🧠 六、总结对比表

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| `linprog` | 易用、标准化接口 | 小中规模 LP 问题 |
| `PuLP` / `CVXPY` | 更高层抽象、更灵活 | 教学/建模优化问题 |
| `Gurobi` / `CPLEX` | 商业级求解器，速度快 | 工业界大规模 LP/IP |

---

## 📌 七、常见错误处理

| 错误 | 原因 | 解决方法 |
|------|------|-----------|
| `success=False` | 无可行解或未收敛 | 检查约束是否冲突，尝试其他 method |
| `infeasible` | 无解 | 放宽约束条件 |
| `unbounded` | 无界解 | 添加更多约束限制变量范围 |

---
