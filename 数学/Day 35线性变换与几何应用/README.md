使用 **矩阵变换**（如仿射变换）可以对图像进行 **旋转、平移、缩放、投影等操作**。这些变换本质上是通过 **矩阵乘法** 对图像的每个像素点坐标进行变换。

在 Python 中，我们可以使用 `NumPy` 和 `OpenCV` 或 `scikit-image` 来实现这些变换。

---

## ✅ 一、图像变换的基本原理

### 🧠 图像变换 = 坐标变换 + 插值

- 每个像素点 $ (x, y) $ 被视为一个二维向量
- 使用变换矩阵 $ M $ 将其映射到新的位置 $ (x', y') $

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
=
M \cdot
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

这种形式称为 **齐次坐标表示法**，便于处理平移、旋转、缩放等操作。

---

## ✅ 二、常用变换矩阵

| 变换类型 | 变换矩阵 |
|----------|----------|
| 平移（Translation） | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ |
| 缩放（Scaling） | $\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |
| 旋转（Rotation） | $\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |
| 投影（Perspective） | 3×3 矩阵（非仿射） |

---

## ✅ 三、Python 实现：使用 OpenCV 进行图像旋转和仿射变换

### 📦 安装依赖：

```bash
pip install opencv-python matplotlib numpy
```

### 🧪 示例代码：图像旋转

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 加载图像并转换为 NumPy 数组
img = cv2.imread('test_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
h, w = img.shape[:2]

# Step 2: 构造旋转变换矩阵
center = (w // 2, h // 2)        # 图像中心
angle = 45                         # 旋转角度
scale = 1.0                        # 缩放比例
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Step 3: 应用仿射变换
rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))

# Step 4: 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title(f"Rotated {angle}°")
plt.imshow(rotated_img)

plt.tight_layout()
plt.show()
```

---

## ✅ 四、示例：自定义仿射变换（例如剪切或平移）

```python
# 定义三个点（原图中的三点）及其变换后的位置
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

# 构建仿射变换矩阵
affine_matrix = cv2.getAffineTransform(pts1, pts2)

# 应用仿射变换
affined_img = cv2.warpAffine(img, affine_matrix, (w, h))

# 显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Affine Transformed")
plt.imshow(affined_img)

plt.tight_layout()
plt.show()
```

---

## ✅ 五、示例：透视变换（Projection / Perspective Transform）

适用于图像“倾斜”或“矫正”，比如拍身份证时的视角变形。

```python
# 定义四个角点（原图中矩形的四个角）
pts1 = np.float32([[135, 50], [300, 50], [130, 200], [310, 200]])

# 目标点（希望这四个点变成的矩形）
pts2 = np.float32([[0, 0], [200, 0], [0, 150], [200, 150]])

# 构建透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

# 应用透视变换
projected_img = cv2.warpPerspective(img, perspective_matrix, (200, 150))

# 显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Perspective Transformed")
plt.imshow(projected_img)

plt.tight_layout()
plt.show()
```

---

## ✅ 六、总结对比表

| 变换类型 | 矩阵维度 | 是否保持平行线 | 是否保持角度 |
|----------|-----------|----------------|--------------|
| 平移     | 3×3       | 是             | 是           |
| 缩放     | 3×3       | 是             | 是           |
| 旋转     | 3×3       | 是             | 是           |
| 仿射变换 | 3×3       | 是             | 否           |
| 透视变换 | 3×3       | 否             | 否           |

---

## ✅ 七、应用场景举例

| 场景 | 变换类型 |
|------|----------|
| OCR 文字识别前的图像矫正 | 透视变换 |
| 图像增强/数据增强 | 旋转、缩放、平移 |
| 图像拼接（Stitching） | 仿射/透视变换 |
| 视频特效（如镜头扭曲） | 自定义变换矩阵 |

---
