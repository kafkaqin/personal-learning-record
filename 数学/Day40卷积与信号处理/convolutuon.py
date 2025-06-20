import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve2d(image,kernel):
    kh,kw=kernel.shape
    ih,iw=image.shape
    pad_h,pad_w=kh//2,kw//2
    padded_img = np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),mode='edge')
    result = np.zeros_like(image)
    for i in range(ih):
        for j in range(iw):
            region = padded_img[i:i+kh,j:j+kw]
            result[i,j] = np.sum(region*kernel)

    return result


img = cv2.imread('img.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gaussian_kernel = np.array(
    [
        [1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,36,24,6],
        [4, 16, 24, 16, 4],
        [1,4,6,4,1]
    ]
)/256.0

blurred_img = cv2.filter2D(gray_img,-1,gaussian_kernel)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(gray_img, cmap='gray')

plt.subplot(1,2,2)
plt.title('Gaussian Kernel')
plt.imshow(blurred_img, cmap='gray')
plt.tight_layout()
plt.savefig('Gaussian_Kernel.png')

blurred_manual=convolve2d(gray_img,gaussian_kernel)
print(blurred_manual.shape)


