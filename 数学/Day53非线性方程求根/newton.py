from scipy.optimize import newton
import numpy as np

def f(x):
    return x**3 - 2*x +5

def df(x):
    return 3*x**2-2

x0 = 2.0

root = newton(f, x0,fprime=df,tol=1e-8,maxiter=50)
print("找到的根 x =",root)
print("验证 f(x) =",f(root))

root = newton(f, x0,tol=1e-8,maxiter=50)
print("找到的根 x =",root)
print("验证 f(x) =",f(root))

import matplotlib.pyplot as plt
x = np.linspace(1.5 , 3.0, 400)
y = f(x)
plt.figure(figsize=(10,6))
plt.plot(x,y,label="$f(x) = x**3-2x-5$")
plt.axhline(0,  color='k',linewidth=0.5)

plt.plot(root,f(root),'ro',label='Root')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("牛顿法求根示意图")
plt.grid(True)
plt.legend()
plt.savefig("f(x).png")
