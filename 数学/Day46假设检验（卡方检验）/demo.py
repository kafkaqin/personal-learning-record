import numpy as np
from scipy.stats import chi2_contingency

data = np.array([[45,15],[30,25]])

chi2, p, dof, expected = chi2_contingency(data)

print("卡方统计量:",chi2)
print("P值:",p)
print("自由度:",dof)
print("期望频率表: \n",expected)

alpha = 0.05

if p > alpha:
    print("拒绝原假设(H0),认为疗法和治疗结果之间存在关联")
else:
    print("无法拒绝原假设(H0),没有足够的证据表明疗法和治疗结果之间有关联.")