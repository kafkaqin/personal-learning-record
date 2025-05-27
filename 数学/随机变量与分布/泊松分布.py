import numpy as np
import matplotlib.pyplot as plt
lam = 5
size = 1000

poisson_data = np.random.poisson(lam=lam, size=size)

# 绘制直方图
plt.hist(poisson_data, bins=np.arange(-0.5, max(poisson_data)+1.5, 1),
         density=True, alpha=0.7, color='orange', edgecolor='black')

plt.title('Poisson Distribution (λ=5)')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.xticks(range(0, 15))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('poisson.png')