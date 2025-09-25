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



import numpy as np
from scipy.stats import chi2_contingency

observed = np.array([
    [60,40],
    [30,70]
])
chi2,p_value,dof,expected=chi2_contingency(observed)
print("观测频数：")
print(observed)
print("\n期望频数：")
print(expected)
print(f"\n卡方统计量: {chi2:.4f}")
print(f"p值: {p_value:.4f}")
print(f"自由度: {dof}")

# from scipy.stats import chisquare

# observed = [8,12,9,11,10,10]
# expected = [10]*6
# chi2,p=chisquare(observed,expected)
# print(f"卡方值: {chi2:.4f}, p值: {p:.4f}")

def cramers_v(chi2,n,r,c):
    return np.sqrt(chi2/(n*min(r-1,c-1)))

n = observed.sum()
r,c = observed.shape
v  = cramers_v(chi2,n,r,c)
print(v)
