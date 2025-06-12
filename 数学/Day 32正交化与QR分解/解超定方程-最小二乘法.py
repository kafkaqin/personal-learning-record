import numpy as np
A = np.array([[2,1],[1,1],[0,1]],dtype=np.float64)

b = np.array([5,3,2],dtype=np.float64)

Q,R = np.linalg.qr(A)

R_upper = R[:2,:]

Qb = (Q.T @ b )[:2]
x_ls = np.linalg.solve(R_upper,Qb)
print("最小二乘解x:",x_ls)