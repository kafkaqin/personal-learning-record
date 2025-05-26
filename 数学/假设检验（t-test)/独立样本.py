from scipy.stats import ttest_ind
import numpy as np

group_a = np.random.normal(loc=75,scale=10,size=30)
group_b = np.random.normal(loc=82,scale=10,size=30)

t_stat,p_value = ttest_ind(group_a,group_b)

print("t 统计量:",t_stat)
print("p 值",p_value)

if p_value < 0.05:
    print("结论：两组数据均值存在显著差异（p < 0.05）")
else:
    print("结论：两组数据均值无显著差异（p ≥ 0.05）")