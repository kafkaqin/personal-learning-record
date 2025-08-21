import numpy as np
import matplotlib.pyplot as plt

study_time = np.array([2,3,5,6,7,8,9,10,12,13])
scores = np.array([55,60,65,70,75,80,85,88,92,95])

cov_matrix = np.cov(study_time,scores)
print("协方差矩阵:",cov_matrix)

convariance = cov_matrix[0,1]
print(f"协方差:{convariance:.2f}")

corr_matrix = np.corrcoef(study_time,scores)
print("关系矩阵:",corr_matrix)
correlation = corr_matrix[0,1]
print(f"相关系数:{correlation:.2f}")

mean_x = np.mean(study_time)
mean_y = np.mean(scores)
manual_cov = np.sum((study_time-mean_x)*(scores-mean_y))/(len(study_time)-1)
print(f"手动计算协方差:{manual_cov:.2f}")

std_x = np.std(study_time,ddof=1)
std_y = np.std(scores,ddof=1)
manual_corr = manual_cov / (std_x * std_y)
print(f"手动计算相关系数:{manual_corr:.2f}")


plt.figure(figsize=(8, 6))
plt.scatter(study_time, scores, color='blue', s=60, alpha=0.8)
plt.title('学习时间 vs 考试成绩')
plt.xlabel('学习时间（小时）')
plt.ylabel('考试成绩')
plt.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(study_time, scores, 1)
p = np.poly1d(z)
plt.plot(study_time, p(study_time), "r--", alpha=0.8, label=f'趋势线 (r={correlation:.3f})')
plt.legend()

plt.savefig('demo1111.png')

sleep_time = [8,7,6,7,6,6,5,5,4,4]
data = np.array([study_time,sleep_time,scores])

cov_all = np.cov(data)
corr_all = np.corrcoef(data)
print(f"多变量协方差:{corr_all:.2f}")
print(f"\n多变量相关系数:{cov_all:.2f}")