import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
np.random.seed(42)
class_A = np.random.normal(78,5,30)
class_B = np.random.normal(82,6,28)

print(f"A.mean={class_A.mean()}, A.std={class_A.std()}")
print(f"B.mean={class_B.mean()}, B.std={class_B.std()}")

t_test,p_value = stats.ttest_ind(class_A,class_B,equal_var=False)
print(f"t_test={t_test}, p_value={p_value}")

alpha = 0.05
if p_value < alpha:
    print("差异")
else:
    print("无差异")

np.random.seed(100)
before = np.random.normal(70,8,20)
after = np.random.normal(5,4,20)
t_stat_paired,p_value_paired=stats.ttest_rel(before,after)
print(f"before.mean{before.mean()}, before.std{before.std()}")
print(f"before.mean{before.mean()}, before.std{before.std()}")
print(f"before.mean-after.mean:{np.mean(after-before)}")

print(f"t_stat_paired={t_stat_paired}, p_value_paired={p_value_paired}")

alpha = 0.05
if p_value_paired < alpha:
    print("有效")
else:
    print("无效")

sample_scores = np.random.normal(78,6,25)
t_test_1,p_value_1= stats.ttest_1samp(sample_scores,75)
print(f"sample_scores.mean={sample_scores.mean()},t_test_1={t_test_1}, p_value_1={p_value_1}")
alpha = 0.05
if p_value_1 < alpha:
    print("显著")
else:
    print("无显著差异")