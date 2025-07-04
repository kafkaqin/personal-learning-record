import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5])
y = np.array([0,1,2,3,4,5])

linear_interp = interp1d(x,y,kind='linear')
quadratic_interp = interp1d(x,y,kind='quadratic')
cubic_interp = interp1d(x,y,kind='cubic')

x_new = np.linspace(0,5,100)

y_linear = linear_interp(x_new)
y_quadratic = quadratic_interp(x_new)
y_cubic = cubic_interp(x_new)

plt.figure(figsize=(10,6))

plt.plot(x,y,'o',label='原始数据点')
plt.plot(x_new,y_linear,'-',label='线性插值')
plt.plot(x_new,y_quadratic,'-',label='二次样条插值')
plt.plot(x_new,y_cubic,'-',label='三次样条插值')
plt.legend()
plt.title("不同插值方法对比")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig("不同插值方法对比.png")