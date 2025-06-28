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