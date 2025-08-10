你提到的是使用 **NumPy**（Python 的科学计算库）来计算矩阵的 **行列式** 和 **逆矩阵**。下面我为你详细说明这两个操作的用法和注意事项。

---

## ✅ 1. 计算矩阵的行列式：`np.linalg.det(A)`

### 作用：
计算方阵 $ A $ 的行列式（determinant）。

### 示例代码：

```python
import numpy as np

# 定义一个 2x2 矩阵
A = np.array([[4, 2],
              [1, 3]])

# 计算行列式
det_A = np.linalg.det(A)
print("行列式 det(A) =", det_A)
```

### 输出：
```
行列式 det(A) = 10.0
```

> ✅ 公式：对于 $ 2 \times 2 $ 矩阵 $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$，行列式为 $ ad - bc = 4*3 - 2*1 = 12 - 2 = 10 $

---

## ✅ 2. 计算矩阵的逆矩阵：`np.linalg.inv(A)`

### 作用：
计算可逆方阵 $ A $ 的逆矩阵 $ A^{-1} $，满足 $ A \cdot A^{-1} = I $（单位矩阵）。

### 示例代码：

```python
# 计算逆矩阵
A_inv = np.linalg.inv(A)
print("逆矩阵 A^(-1) =\n", A_inv)
```

### 输出：
```
逆矩阵 A^(-1) =
[[ 0.3  -0.2 ]
 [-0.1   0.4 ]]
```

> ✅ 验证：`np.dot(A, A_inv)` 应该接近单位矩阵。

```python
print("A @ A_inv =\n", A @ A_inv)
```

输出应为近似单位矩阵：
```
[[1. 0.]
 [0. 1.]]
```

---

## ⚠️ 注意事项

| 问题 | 说明 |
|------|------|
| **矩阵必须是方阵** | 只有 $ n \times n $ 的矩阵才能计算行列式和逆矩阵 |
| **行列式不能为 0** | 如果 `det(A) ≈ 0`，矩阵是**奇异的**（不可逆） |
| **数值精度问题** | `np.linalg.det()` 返回的是浮点数，判断是否为 0 时建议用 `np.isclose(det_A, 0)` |

---

## ✅ 完整示例：判断是否可逆并求逆

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

det_A = np.linalg.det(A)

if np.isclose(det_A, 0):
    print("矩阵不可逆（行列式为 0）")
else:
    print(f"行列式 = {det_A:.2f}")
    A_inv = np.linalg.inv(A)
    print(f"逆矩阵 =\n{A_inv}")
```

---

## 🧮 其他常用线性代数操作（NumPy）

| 功能 | 函数 |
|------|------|
| 矩阵乘法 | `A @ B` 或 `np.dot(A, B)` |
| 转置 | `A.T` |
| 特征值和特征向量 | `np.linalg.eig(A)` |
| 矩阵的秩 | `np.linalg.matrix_rank(A)` |
| 解线性方程 $ Ax = b $ | `np.linalg.solve(A, b)` |

---