import numpy as np
mu = 0
sigma = 1
size = 1000
normal_data = np.random.normal(mu, sigma, size)
# print(normal_data)

import matplotlib.pyplot as plt
from scipy.stats import norm

# 绘制直方图
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# 添加拟合的正态分布曲线
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r', label='PDF')

plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('normal_distribution.png')