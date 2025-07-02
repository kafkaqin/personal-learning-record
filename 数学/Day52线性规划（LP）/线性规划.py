from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
c = [-3,-5]

A_ub = [
     [3,2],
     [1,0],
     [0,1],
 ]
b_ub = [18,4,6]

bounds = [(0,None),(0,None)]

result = linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')

print("是否找到最优解:",result.success)
print("最优值:",-result.fun)
print("最优解x1,x2:",result.x)

x1 = np.linspace(0,5,400)
x2 = np.linspace(0,7,400)
X1,X2 = np.meshgrid(x1,x2)

plt.figure(figsize=(8,6))
plt.fill_betweenx(x2,0,(18-2*x2)/3,where=(x2<=6),color='red',alpha=0.1,label='3x1+2x2 <= 18')
plt.fill_between(x1,0,6,where=(x1<=4),color='blue',alpha=0.1,label='x2 <= 6 and x1<=4')

Z = 3*X1+5*x2
CS = plt.contour(X1,X2,Z,colors='green',linestyles='dashed')
plt.clabel(CS, inline=True, fontsize=8)
plt.plot(2,6,'ro',label='Optimal Solution (2,6)')

plt.xlim(0,5)
plt.ylim(0,7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("线性规划可行域和最优解")
plt.legend()
plt.grid(True)
plt.savefig('线性规划可行域和最优解.png')