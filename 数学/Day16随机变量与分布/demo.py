import numpy as np
import matplotlib.pyplot as plt

# 参数设置
mu = 100      # 均值
sigma = 15    # 标准差
size = 1000   # 生成样本数量

# 生成正态分布数据（如 IQ 分数模拟）
normal_data = np.random.normal(loc=mu, scale=sigma, size=size)

# 查看统计信息
print("正态分布数据统计：")
print(f"均值: {normal_data.mean():.2f}")
print(f"标准差: {normal_data.std():.2f}")
print(f"最小值: {normal_data.min():.2f}, 最大值: {normal_data.max():.2f}")

# 可视化
plt.figure(figsize=(9, 5))
plt.hist(normal_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='生成数据')

# 绘制理论概率密度函数（PDF）
from scipy.stats import norm
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='理论PDF')
plt.title('正态分布数据 (μ=100, σ=15)')
plt.xlabel('值')
plt.ylabel('密度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("normal_data.png")

# 参数设置
lam = 3      # 平均每单位时间发生 3 次事件
size = 1000

# 生成泊松分布数据
poisson_data = np.random.poisson(lam=lam, size=size)

# 查看统计信息
print("\n泊松分布数据统计：")
print(f"均值: {poisson_data.mean():.2f}（理论: {lam}）")
print(f"方差: {poisson_data.var():.2f}（理论: {lam}）")
print(f"唯一值:", np.unique(poisson_data))

# 可视化
plt.figure(figsize=(9, 5))
values, counts = np.unique(poisson_data, return_counts=True)
plt.vlines(values, 0, counts / size, colors='b', lw=3, alpha=0.8, label='模拟频率')

# 理论概率质量函数（PMF）
from scipy.stats import poisson
x = np.arange(0, max(poisson_data)+1)
theoretical_pmf = poisson.pmf(x, lam)
plt.plot(x, theoretical_pmf, 'ro', markersize=6, label='理论PMF')

plt.title(f'泊松分布数据 (λ={lam})')
plt.xlabel('事件发生次数 k')
plt.ylabel('概率 P(X=k)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("poisson_data.png")