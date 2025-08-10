import numpy as np
A = np.array([[4,2],
              [1,3]])

b = np.array([14,8])

x = np.linalg.solve(A,b)
print(x)

print(A @ x)

import numpy as np

A = np.array([[4, 2],
              [1, 3]])
b = np.array([14, 8])

try:
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        print("矩阵 A 不可逆（行列式为 0），无法求解")
    else:
        x = np.linalg.solve(A, b)
        print("解为：x =", x)
except np.linalg.LinAlgError as e:
    print("求解失败：", e)