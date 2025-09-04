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


import numpy as np
A = np.array(
    [
        [2,1,1],
        [1,3,2],
        [1,0,0],
    ],dtype=float,
)

b = np.array([5,7,2],dtype=float)

Q,R = np.linalg.qr(A)
print(f"分解正确?{np.allclose(A,Q@R)}")

Qb = Q.T @ b
x = np.linalg.solve(R,Qb)
print("解x=",x)

Ax = A@x
print(f"正确?{np.allclose(Ax,b)}")

def back_substitution(R,Qb):
    n = len(Qb)
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (Qb[i]-np.sum(R[i,i+1:]*x[x+1:]))/R[i,i]
    return x
# x_manual = back_substitution(R,Qb)
# print(f"手动实现正确:{np.allclose(x,x_manual)}")

x_qr = np.linalg.solve(R,Q.T@b)
x_inv = np.linalg.inv(A) @b
x_solve = np.linalg.solve(A,b)

x_data = np.array([1,2,3,4])
y_data = np.array([1.5,3.1,4.8,6.3])

A_ls = np.column_stack([x_data,np.ones_like(x_data)])
b_ls = y_data
Q_ls,R_ls = np.linalg.qr(A_ls)
Qb_ls = Q_ls.T @ b_ls

R_eff = R_ls[:2,:]
Qb_eff = Qb_ls[:2]
coeffs = np.linalg.solve(R_eff,Qb_eff)
a,b = coeffs
a, b = coeffs
print(f"拟合直线: y = {a:.2f}x + {b:.2f}")

# 可视化
import matplotlib.pyplot as plt
plt.scatter(x_data, y_data, color='red', label='数据点')
plt.plot(x_data, a*x_data + b, label=f'拟合线 y={a:.2f}x+{b:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Qr_R.png")