import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 5, 400)
C = 2*x**2-8*x+10

plt.figure(figsize=(8, 5))
plt.plot(x, C, label=r'$C(x) = 2x^2 - 8x + 10$')
plt.scatter(2, 2, color='red', zorder=5, label='最小成本点 (2, 2)')
plt.xlabel('生产数量 x（百件）')
plt.ylabel('成本 C(x)（万元）')
plt.title('成本函数最小化')
plt.legend()
plt.grid(True)
plt.savefig("C.png")