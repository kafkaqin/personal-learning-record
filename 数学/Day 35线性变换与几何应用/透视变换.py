import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]

pts1 = np.float32([[135, 50], [300, 50], [130, 200],[310, 200]])
pts2 = np.float32([[0, 0], [200, 0], [0, 150],[200, 150]])

perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
projected_img = cv2.warpPerspective(img, perspective_matrix, (200, 150))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.savefig('test11-original-image.png')

plt.subplot(1, 2, 2)
plt.title('Affine Image')
plt.savefig('test11-projected-image.png')

plt.tight_layout()
