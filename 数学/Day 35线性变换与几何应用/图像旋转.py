import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]

center = (w // 2, h // 2)
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotation_img = cv2.warpAffine(img, rotation_matrix, (w, h))
plt.subplot(1, 2, 2)
plt.title(f"Rotation {angle}")
plt.imshow(rotation_img)
plt.tight_layout()
plt.savefig(f"Rotation_{angle}.png")

