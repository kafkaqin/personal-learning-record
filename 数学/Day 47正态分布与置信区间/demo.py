import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

np.random.seed(42)
sample = np.random.normal(loc=100,scale=15,size=100)
sample_mean = np.mean(sample)
sample_std = np.std(sample)
n = len(sample)

se = sample_std/np.sqrt(n)

condidence_level = 0.95
ci = norm.interval(condidence_level,loc=sample_mean,scale=se)
print(f"样本均值:{sample_mean:.2f}")
print(f"{int(condidence_level*100)}% 置信区间：{ci}")

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------
# 参数设置
# -----------------------------
np.random.seed(42)
n = 30          # 样本量
mu_true = 100   # 真实均值（未知）
sigma = 15      # 已知总体标准差
confidence_level = 0.95  # 置信水平

# 生成样本
sample = np.random.normal(loc=mu_true, scale=sigma, size=n)
x_bar = np.mean(sample)  # 样本均值

# -----------------------------
# 计算置信区间（Z 区间）
# -----------------------------
alpha = 1 - confidence_level
# 使用 norm.interval 计算
ci_lower, ci_upper = norm.interval(confidence_level,
                                   loc=x_bar,
                                   scale=sigma/np.sqrt(n))

print(f"样本均值: {x_bar:.2f}")
print(f"{int(confidence_level*100)}% 置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")