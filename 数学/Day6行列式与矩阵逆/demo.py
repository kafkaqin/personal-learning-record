import  numpy as np
A = np.array([[4,2],
              [1,3]])

det_A = np.linalg.det(A)
print(det_A)

A_inv = np.linalg.inv(A)
print(A_inv)
s = np.dot(A_inv, A)
print(s)