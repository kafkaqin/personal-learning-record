使用 **卷积（Convolution）** 是图像处理中最基础且强大的技术之一，可以实现各种滤波效果，如：

- 高斯模糊（Gaussian Blur）
- 锐化（Sharpen）
- 边缘检测（Edge Detection）
- 均值模糊（Mean Filter）

---

## ✅ 一、基本原理：图像卷积

图像卷积就是将一个 **小矩阵（称为卷积核或滤波器 kernel）** 在图像上滑动，并与对应区域进行逐元素相乘后求和，生成一个新的像素值。

$$
\text{Output}(i,j) = \sum_{m}\sum_{n} \text{Image}(i+m, j+n) \cdot \text{Kernel}(m,n)
$$

---

## ✅ 二、Python 实现：用 NumPy 和 OpenCV 实现高斯模糊

### 📦 安装依赖：

```bash
pip install opencv-python numpy matplotlib
```

---

## 🧪 三、示例代码：使用卷积实现高斯模糊

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 读取图像
img = cv2.imread('test_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰度图

# Step 2: 定义高斯滤波核（5x5）
gaussian_kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
]) / 256.0  # 归一化

# Step 3: 使用 OpenCV 的 filter2D 进行卷积操作
blurred_img = cv2.filter2D(gray_img, -1, gaussian_kernel)

# Step 4: 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(gray_img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("高斯模糊图像")
plt.imshow(blurred_img, cmap='gray')

plt.tight_layout()
plt.show()
```

---

## ✅ 四、其他常见滤波核（kernel）

你可以替换上面的 `gaussian_kernel` 来实现不同滤波效果：

| 滤波类型 | 卷积核 |
|----------|--------|
| 均值模糊（Blur） | `np.ones((5,5))/25` |
| 锐化（Sharpen） | `[[ 0, -1,  0], [-1, 5, -1], [ 0, -1,  0]]` |
| Sobel 边缘检测 X 方向 | `[[-1,0,1],[-2,0,2],[-1,0,1]]` |
| Sobel 边缘检测 Y 方向 | `[[-1,-2,-1],[0,0,0],[1,2,1]]` |
| Laplacian 边缘检测 | `[[0,1,0],[1,-4,1],[0,1,0]]` |

---

## ✅ 五、手动实现二维卷积函数（可选）

如果你不想使用 OpenCV，也可以手动实现卷积函数（仅用于学习目的）：

```python
def convolve2d(image, kernel):
    kh, kw = kernel.shape
    ih, iw = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    result = np.zeros_like(image)
    for i in range(ih):
        for j in range(iw):
            region = padded_img[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result
```

然后这样调用：

```python
blurred_manual = convolve2d(gray_img, gaussian_kernel)
```

---

## ✅ 六、应用场景举例

| 应用 | 描述 |
|------|------|
| 图像去噪 | 使用均值/高斯滤波平滑噪声 |
| 物体识别 | 提取边缘特征作为预处理步骤 |
| 视频特效 | 实时应用各种滤镜 |
| 医学图像增强 | 改善图像对比度、细节清晰度 |

---