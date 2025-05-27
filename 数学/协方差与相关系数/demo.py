import numpy as np

a = np.array([1, 2, 3,4,5])
b = np.array([6,7,8,9,10])
cov_matrix = np.cov(a,b)
print("协方差矩阵:\n",cov_matrix)

cov_ab = cov_matrix[0,1]
print("协方差Cov(a,b):",cov_ab)

corr_matrix = np.corrcoef(a,b)
print("相关系数矩阵:\n",corr_matrix)

import matplotlib.pyplot as plt

plt.scatter(a, b, color='blue')
plt.title('Scatter Plot of a vs b')
plt.xlabel('a')
plt.ylabel('b')
plt.grid(True)
plt.savefig("Scatter.png")
corr_ab = corr_matrix[0,1]
print("相关系数r(a,b):",corr_ab)