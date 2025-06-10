使用 **奇异值分解（SVD）** 进行图像压缩是一种经典的降维方法，尤其适用于灰度图像。其基本思想是：将图像矩阵进行 SVD 分解，保留前 k 个最大的奇异值和对应的向量，重构图像以实现压缩。

---

## ✅ 一、原理简述

对于一个 $ m \times n $ 的图像矩阵 $ A $，我们可以对其进行奇异值分解：

$$
A = U \Sigma V^T
$$

其中：

- $ U $ 是 $ m \times m $ 正交矩阵（左奇异向量）
- $ \Sigma $ 是 $ m \times n $ 对角矩阵，对角线元素为奇异值（从大到小排列）
- $ V^T $ 是 $ n \times n $ 正交矩阵的转置（右奇异向量）

如果我们只保留前 $ k $ 个最大的奇异值及其对应的左右向量，可以近似重构图像：

$$
A_{\text{approx}} = U_k \Sigma_k V_k^T
$$

这样就能在保留主要信息的同时减少存储空间，达到图像压缩效果。

---

## ✅ 二、Python 示例代码（使用 NumPy）

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: 加载图像并转换为灰度图
def load_image(path):
    img = Image.open(path).convert('L')  # 'L' 表示灰度图
    img_matrix = np.array(img, dtype=np.float64)
    return img_matrix

# Step 2: 使用 SVD 压缩图像
def compress_image_svd(image_matrix, k):
    U, s, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return compressed

# Step 3: 显示原始与压缩后的图像
def show_images(original, compressed, k):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed (k={k})")
    plt.imshow(compressed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 主程序流程
if __name__ == "__main__":
    # 图像路径（请替换为你自己的图片路径）
    image_path = "test_image.jpg"
    
    # 加载图像
    original_img = load_image(image_path)

    # 设置保留的奇异值数量
    k = 30

    # 压缩图像
    compressed_img = compress_image_svd(original_img, k)

    # 显示图像
    show_images(original_img, compressed_img, k)
```

---

## ✅ 三、输出说明

- `original_img`: 原始图像矩阵（float64 类型）
- `compressed_img`: 用前 k 个奇异值重建的图像矩阵
- 可视化显示原始图像 vs 压缩后的图像

> ⚠️ 注意：
> - 越大的 `k` 值压缩质量越高，但占用空间也越大。
> - `k=0~min(m,n)`，通常选择 `k << min(m,n)` 实现压缩。

---

## 📌 四、压缩率计算（可选）

我们可以通过比较原始数据和压缩后所需的数据大小来估算压缩比：

```python
def compression_ratio(m, n, k):
    original_size = m * n
    compressed_size = k * (m + n + 1)  # U_k(k×m), s(k), V_k(n×k)
    ratio = original_size / compressed_size
    return ratio

m, n = original_img.shape
ratio = compression_ratio(m, n, k)
print(f"压缩率（原始大小 / 压缩后大小）≈ {ratio:.2f}")
```

---

## ✅ 五、应用场景举例

| 场景 | 说明 |
|------|------|
| 图像压缩 | 减少图像存储空间或传输带宽 |
| 数据降维 | 用于机器学习中特征提取 |
| 噪声去除 | 通过舍弃较小奇异值过滤噪声 |
| 面部识别 | Eigenfaces 方法的基础 |

---

## 🧩 六、拓展建议

- 支持彩色图像（RGB 三个通道分别处理）
- 将压缩结果保存为新图像文件
- 使用 `scikit-learn` 中的 `TruncatedSVD` 处理稀疏数据
- 使用 PCA 等价于 SVD 在协方差矩阵上的应用

---

如果你有一张具体的图像文件（如 JPG 或 PNG），我可以帮你写出完整的压缩流程，包括读取、压缩、可视化、保存