import numpy as np

image = np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8)
print("\n RGB图像数组:\n",image)

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]
print("\n 红色通道:\n",red_channel)
print("\n 绿色通道:\n",green_channel)
print("\n 蓝色通道:\n",blue_channel)