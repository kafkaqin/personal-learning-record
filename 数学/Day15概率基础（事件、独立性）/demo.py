import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

n = 10
p = 0.5
N = 10000
heads_counts = np.random.binomial(n=n, p=p, size=N)
print(heads_counts[:10])
mean_sim = np.mean(heads_counts)
var_sim = np.var(heads_counts)

print(f"\n模拟结果:")
print(f"平均值（期望）: {mean_sim:.4f} （理论值: {n*p:.1f}）")
print(f"方差: {var_sim:.4f} （理论值: {n*p*(1-p):.1f}）")

# 绘制直方图
plt.figure(figsize=(9, 5))
counts, bins, patches = plt.hist(heads_counts, bins=np.arange(-0.5, n+1.5, 1),
                                 density=True, alpha=0.7, color='skyblue', edgecolor='black', label='模拟结果')

# 理论概率质量函数（PMF）—— 二项分布
from scipy.stats import binom
x = np.arange(0, n+1)
theoretical_pmf = binom.pmf(x, n, p)
plt.plot(x, theoretical_pmf, 'ro-', label='理论分布', linewidth=2)

plt.title(f'抛硬币实验模拟（{N} 次实验，每次 {n} 次抛掷）')
plt.xlabel('正面出现次数')
plt.ylabel('概率')
plt.xticks(range(0, n+1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("de.png")

single_toss = np.random.binomial(1,0.5)
print(single_toss)
biased_heads = np.random.binomial(n=10, p=0.7, size=1000)  # 有偏硬币
print(f"不公平硬币平均正面数: {np.mean(biased_heads):.2f}")

prob_at_least_6 = np.mean(heads_counts >= 6)
print(f"至少出现6次正面的概率（模拟）: {prob_at_least_6:.4f}")

# 理论值（使用 scipy）
theoretical_prob = 1 - binom.cdf(5, n, p)
print(f"理论概率: {theoretical_prob:.4f}")

prob_at_least_6 = np.mean(heads_counts >= 6)
print(f"至少出现6次正面的概率（模拟）: {prob_at_least_6:.4f}")

# 理论值（使用 scipy）
theoretical_prob = 1 - binom.cdf(5, n, p)
print(f"理论概率: {theoretical_prob:.4f}")