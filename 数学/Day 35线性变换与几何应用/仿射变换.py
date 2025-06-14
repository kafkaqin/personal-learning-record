import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

affine_matrix = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(img, affine_matrix, (h, w))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.savefig('test11-original-image.png')

plt.subplot(1, 2, 2)
plt.title('Affine Image')
plt.savefig('test11-affine-image.png')

plt.tight_layout()
