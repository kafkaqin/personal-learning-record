import numpy as np

A = np.array(
    [
    [3,2],
    [1,-1]
],dtype=np.float64)

b = np.array([1,2],dtype=np.float64)

Q,R = np.linalg.qr(A)
print(Q)
print(R)
Qb = Q.T @ b

x = np.linalg.solve(R,Qb)
print("解x",x)