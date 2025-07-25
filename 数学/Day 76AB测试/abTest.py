import numpy as np
from scipy import  stats

n_A = 5000
x_A = 485
n_B = 5000
x_B = 525

p_A = x_A/n_A
p_B = x_B/n_B

p_pooled = (x_A + x_B)/(n_A + n_B)

SE = np.sqrt(p_pooled * (1 - p_pooled)*(1/n_A+1/n_B))

diff = p_B -p_A

z = stats.norm.ppf(0.975)
ci_lower = diff - z * SE
ci_upper = diff + z * SE

print(f"转化率 A: {p_A:.4f}")
print(f"转化率 B: {p_B:.4f}")
print(f"转化率差异 : {diff:.4f}")
print(f"95% 置信区间: {ci_lower:.4f}, {ci_upper:.4f}")

if ci_lower > 0  or  ci_upper < 0:
    print("差异显著 (置信区间不包含 0)")
else:
    print("差异不显著 (置信区间包含 0)")