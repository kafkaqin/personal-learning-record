使用 `numpy.ndarray` 是处理**高维数据（如图像、张量、多维数组）** 的核心方式。NumPy 提供了强大的 **n 维数组对象 `ndarray`**，可以轻松地表示和操作任意维度的数据。

---

## ✅ 一、什么是 `numpy.ndarray`

`numpy.ndarray` 是 NumPy 中的核心数据结构，支持：

- 多维数组（1D、2D、3D、4D……）
- 同构类型（所有元素类型相同）
- 快速的向量化运算

---

## ✅ 二、创建高维 `ndarray`

```python
import numpy as np

# 1D 数组（向量）
vec = np.array([1, 2, 3])
print("1D 数组:", vec)

# 2D 数组（矩阵）
mat = np.array([[1, 2], [3, 4]])
print("\n2D 数组:\n", mat)

# 3D 数组（张量）
tensor_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("\n3D 数组:\n", tensor_3d)

# 使用 reshape 创建高维数组
arr = np.arange(24).reshape((2, 3, 4))  # 2块，每块3行4列
print("\nreshape 创建的 3D 数组:\n", arr)
```

---

## ✅ 三、ndarray 属性介绍

```python
print("形状（shape）:", arr.shape)      # 输出 (2, 3, 4)
print("维度数（ndim）:", arr.ndim)     # 输出 3
print("元素总数（size）:", arr.size)   # 输出 24
print("数据类型（dtype）:", arr.dtype) # 输出 int64
```

---

## ✅ 四、索引与切片操作（适用于任何维度）

### 🧪 示例：访问 3D 数组中的元素

```python
# arr[块索引][行索引][列索引]
print("arr[0, 1, 2] =", arr[0, 1, 2])  # 输出 6

# 切片：获取第一个块的所有行和列
print("\narr[0, :, :] =\n", arr[0, :, :])

# 切片：获取第2个块的前两行，前两列
print("\narr[1, :2, :2] =\n", arr[1, :2, :2])
```

---

## ✅ 五、常用操作

### 📌 1. 广播（Broadcasting）

```python
# 对整个数组进行标量运算
arr_add = arr + 10
print("加10后的数组:\n", arr_add)
```

### 📌 2. 沿轴操作（axis）

```python
# 计算沿 axis=0（按块合并）的和
sum_axis0 = np.sum(arr, axis=0)
print("\n沿 axis=0 求和:\n", sum_axis0)

# 沿 axis=1（按行求和）
sum_axis1 = np.sum(arr, axis=1)
print("\n沿 axis=1 求和:\n", sum_axis1)
```

### 📌 3. 转置（transpose）

```python
# 将 shape 为 (2,3,4) 的数组转置为 (4,3,2)
transposed = arr.transpose((2, 1, 0))
print("\n转置后的 shape:", transposed.shape)
```

---

## ✅ 六、应用示例：图像数据处理（RGB 图像）

图像在计算机中通常表示为 3D `ndarray`，例如：

- 形状为 `(height, width, channels)`
- `channels` 表示 RGB 三个通道

```python
# 假设一个 3x3 的 RGB 图像（值范围 0~255）
image = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
print("\nRGB 图像数组:\n", image)

# 提取红色通道
red_channel = image[:, :, 0]
print("\n红色通道:\n", red_channel)
```

---

## ✅ 七、高维数据应用场景

| 场景 | 数据维度 | 示例 |
|------|----------|------|
| 图像处理 | 3D/4D | `(H, W, C)` 或 `(N, H, W, C)` |
| 时间序列 | 3D | `(样本数, 时间步, 特征数)` |
| 视频数据 | 5D | `(视频数量, 帧数, 高, 宽, 通道)` |
| 医学影像 | 3D/4D | CT/MRI 扫描体积数据 |

---

## ✅ 八、性能优势

- **向量化操作**：避免使用 for 循环，直接对整个数组执行数学运算
- **内存效率**：存储紧凑，适合大规模数据
- **与科学计算库集成**：如 Pandas、Scikit-learn、TensorFlow、PyTorch 等都基于 `ndarray`

---

## 🧠 小提示

- 使用 `.reshape()` 可以改变数据维度而不改变内容
- 使用 `.squeeze()` 和 `.expand_dims()` 可以删除或增加维度
- 使用 `.astype()` 可以转换数据类型（如 float32 → uint8）

---