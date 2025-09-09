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


import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('2.jpg')
if img is None:
    img = np.zeros((200,200,3),dtype=np.uint8)
    cv2.rectangle(img,(50,50),(150,150),(255,255,255),-1)
else:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(f"图像形状:{img.shape}")
# 显示原图
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 4, 1)
# plt.imshow(img)
# plt.title("原始图像")
# plt.axis('off')
h,w = img.shape[:2]
center = (w // 2, h // 2)
angle = 30
M_rotate = cv2.getRotationMatrix2D(center,angle,1.0)
rotated = cv2.warpAffine(img,M_rotate,(w,h))

src_points = np.float32([[50,50],[180,40],[40,180],[190,190]])
dst_points = np.float32([[40,40],[160,40],[40,160],[160,160]])
M_perspective = cv2.getPerspectiveTransform(src_points,dst_points)
wraped = cv2.warpPerspective(rotated,M_perspective,(w,h))
img_with_points = img.copy()
for pt in src_points:
    cv2.circle(img_with_points,(int(pt[0]),int(pt[1])),5,(255,0,0),-1)

def affine_transform(img,M_2x3):
    h,w = img.shape[:2]
    new_img = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            src_x,src_y = M_2x3 @ np.array([x,y,1])
            src_x,src_y = int(src_x),int(src_y)
            if 0 <=src_x < w and 0 <=src_y < h:
                new_img[y,x] = img[src_y,src_x]
    return new_img

M_rotate = cv2.getRotationMatrix2D(center,30,1.0)
tx,ty = 50,30
M_translate = np.array([
    [1,0,tx],
    [0,1,ty],
],dtype=float)

M_rotate_3x3 = np.vstack([M_rotate,[0,0,1]])
M_translate_3x3 = np.vstack([M_translate,[0,0,1]])
M_combined_3x3 = M_translate_3x3 @ M_rotate_3x3
M_combined = M_combined_3x3[:2,:]
transformed = cv2.warpAffine(img,M_combined,(w,h))
