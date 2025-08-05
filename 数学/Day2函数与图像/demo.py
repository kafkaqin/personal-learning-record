import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei','Arial Unicode MS','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
x = np.linspace(-2*np.pi,2*np.pi,1000)
y_sin = np.sin(x)
y_exp = np.exp(x/4)

plt.figure(figsize=(10,6))
plt.plot(x,y_sin,label='sin(x)',color='blue',linewidth=2)
plt.plot(x,y_exp,label='exp(x/4)',color='red',linewidth=2)
plt.title('正弦函数和指数函数图像',fontsize=16)
plt.xlabel('x',fontsize=12)
plt.ylabel('y',fontsize=12)
plt.grid(True,linestyle='--',alpha=0.6)

plt.legend(fontsize=12)

plt.xticks( [-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
    [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

plt.tight_layout()
plt.savefig("demo.png",dpi=300)