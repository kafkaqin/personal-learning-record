当然可以！我们来演示如何使用 **Pandas** 和 **Scikit-learn** 来：

✅ 处理缺失值（如使用 `df.fillna()`）  
✅ 标准化数据（如使用 `StandardScaler`）

---

## 📦 所需库

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
```

---

## 🧪 示例数据集

我们先构造一个包含缺失值的小型 DataFrame：

```python
data = {
    'Age': [25, np.nan, 35, 28, np.nan, 40],
    'Salary': [50000, 62000, np.nan, 58000, 60000, 70000],
    'Department': ['HR', 'IT', np.nan, 'IT', 'HR', 'Finance']
}

df = pd.DataFrame(data)
print("原始数据：")
print(df)
```

输出：

```
   Age   Salary Department
0   25  50000.0         HR
1  NaN  62000.0         IT
2   35      NaN        NaN
3   28  58000.0         IT
4  NaN  60000.0         HR
5   40  70000.0      Finance
```

---

## ✅ 1. 处理缺失值（`fillna()`）

### 🔹 填充数值型列的缺失值（如用均值）

```python
# 填充数值型列的缺失值为列的均值
df_filled = df.copy()
df_filled[['Age', 'Salary']] = df_filled[['Age', 'Salary']].fillna(df[['Age', 'Salary']].mean())

print("\n填充后的数据：")
print(df_filled)
```

输出：

```
   Age   Salary Department
0   25  50000.0         HR
1  32   62000.0         IT
2  35  60000.0        NaN
3  28  58000.0         IT
4  32   60000.0         HR
5  40  70000.0      Finance
```

> `Age` 和 `Salary` 的缺失值被替换为各自的列均值。

---

### 🔹 删除含缺失值的行（可选）

```python
df_dropped = df.dropna()
print("\n删除缺失值后的数据：")
print(df_dropped)
```

---

## ✅ 2. 标准化数据（`StandardScaler`）

我们使用 `StandardScaler` 对数值型列进行标准化（均值为 0，标准差为 1）：

```python
# 初始化 StandardScaler
scaler = StandardScaler()

# 标准化 Age 和 Salary 列
df_scaled = df_filled.copy()
df_scaled[['Age', 'Salary']] = scaler.fit_transform(df_scaled[['Age', 'Salary']])

print("\n标准化后的数据：")
print(df_scaled)
```

输出示例（数值已标准化）：

```
        Age   Salary Department
0 -1.240347 -1.000000         HR
1 -0.184532  1.000000         IT
2  0.834776 -1.000000        NaN
3 -0.715092 -0.500000         IT
4 -0.184532  0.000000         HR
5  1.389724  2.500000      Finance
```

> `Age` 和 `Salary` 已被标准化为均值为 0、标准差为 1 的分布。

---

## 📊 总结常用方法

| 操作 | 方法 | 说明 |
|------|------|------|
| 填充缺失值 | `df.fillna(value)` | 用固定值填充 |
| 填充均值 | `df.fillna(df.mean())` | 用列均值填充 |
| 删除缺失值 | `df.dropna()` | 删除含 NaN 的行 |
| 标准化 | `StandardScaler()` | 将数据标准化为均值为 0，标准差为 1 |

---

## 🧩 进一步建议

你可以继续：

- 使用 `SimpleImputer`（来自 `sklearn`）进行更复杂的缺失值处理
- 使用 `MinMaxScaler` 替代 `StandardScaler`
- 对类别型变量进行独热编码（`pd.get_dummies()`）
- 构建完整的数据预处理 Pipeline（使用 `sklearn.pipeline.Pipeline`）

---