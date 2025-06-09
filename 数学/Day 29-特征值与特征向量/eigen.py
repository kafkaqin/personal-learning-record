import numpy as np

A = np.array([[4, 2],[1,3]])
print(A)
eigen_values,eigen_vectors = np.linalg.eig(A)
print("特征值:",eigen_values)
print("特征向量:",eigen_vectors)
a = eigen_values[0]
v = eigen_vectors[:,0]

Av = A @ v
av = a * v
print("A @ v:",Av)
print("a * v:",av)
print(av==Av)