当然可以！我们将使用 **Matplotlib** 来绘制：

✅ 散点图（`plt.scatter(x, y)`）  
✅ 热力图（Heatmap）

---

## 📦 所需库

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 用于热力图（可选）
```

---

## ✅ 1. 使用 `plt.scatter()` 绘制散点图

### 📌 示例：随机生成数据并绘制

```python
# 生成随机数据
np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', label='数据点', alpha=0.7)
plt.title('散点图示例')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 🧩 可选：根据第三维变量着色（颜色映射）

```python
colors = np.random.rand(100)  # 第三个维度，用于颜色

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=colors, cmap='viridis', alpha=0.7)
plt.title('带颜色映射的散点图')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.colorbar(scatter, label='颜色映射值')
plt.grid(True)
plt.show()
```

---

## ✅ 2. 使用 Matplotlib 和 Seaborn 绘制热力图（Heatmap）

### 📌 示例 1：使用随机矩阵绘制热力图

```python
# 创建一个 5x5 的随机矩阵
data = np.random.rand(5, 5)

# 使用 Matplotlib + imshow
plt.figure(figsize=(6, 6))
heatmap = plt.imshow(data, cmap='hot', interpolation='nearest')
plt.title('热力图（imshow）')
plt.colorbar(heatmap, label='数值大小')
plt.xticks(np.arange(5), ['A', 'B', 'C', 'D', 'E'])
plt.yticks(np.arange(5), ['1', '2', '3', '4', '5'])
plt.show()
```

---

### 📌 示例 2：使用 Seaborn 绘制更美观的热力图（推荐）

```python
# 创建一个 DataFrame
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'], index=['1', '2', '3', '4', '5'])

plt.figure(figsize=(6, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Seaborn 热力图')
plt.show()
```

---

## 📊 示例数据：鸢尾花数据集（Iris）的热力图（相关系数矩阵）

```python
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# 计算相关系数矩阵
corr = df_iris.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Iris 数据集特征相关性热力图')
plt.show()
```

---

## 📋 总结常用绘图方法

| 图表类型 | 方法 | 说明 |
|----------|------|------|
| 散点图 | `plt.scatter(x, y)` | 显示两个变量之间的关系 |
| 热力图 | `plt.imshow()` / `sns.heatmap()` | 显示矩阵形式数据的强度分布 |
| 颜色映射 | `cmap` 参数 | 设置颜色渐变 |
| 颜色条 | `plt.colorbar()` | 显示颜色对应数值的图例 |
| 注释 | `annot=True` | 在热力图中显示数值 |

---

## 🧩 进一步建议

你可以继续：

- 使用 `plt.hexbin(x, y)` 绘制六边形箱图（适合大数据集）
- 在热力图中使用聚类（如 `sns.clustermap()`）
- 将散点图与颜色映射结合，用于机器学习结果可视化
- 使用子图（`plt.subplots()`）展示多个图表

---