import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data,columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target
print("数据的形状:",df.shape)
print("前5行数据：")
print(df.head())
print("\n描述性统计：")
print(df.describe())
print(f"\n缺失值:\n{df.isnull().sum()}")


plt.figure(figsize=(8, 5))
sns.histplot(df['MedHouseVal'], kde=True, bins=50, color='skyblue')
plt.title('房价中位数分布')
plt.xlabel('房价（$100,000）')
plt.ylabel('频次')
plt.grid(True, alpha=0.3)
plt.savefig("房价分布.png")

sns.pairplot(df[['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal']].sample(1000),
             plot_kws={'alpha': 0.6}, diag_kws={'bins': 30})
plt.suptitle('关键特征与房价关系', y=1.02)
plt.savefig("关键特征与房价关系.png")


plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性热力图')
plt.savefig("特征相关性热力图.png")


from scipy import stats

# 定义高收入区域（中位收入 > 4）
high_income = df[df['MedInc'] > 4]['MedHouseVal']
low_income = df[df['MedInc'] <= 4]['MedHouseVal']

# 独立样本 t 检验
t_stat, p_value = stats.ttest_ind(high_income, low_income, equal_var=False)

print(f"高收入区房价均值: {high_income.mean():.3f}")
print(f"低收入区房价均值: {low_income.mean():.3f}")
print(f"t检验结果: t={t_stat:.4f}, p={p_value:.2e}")

alpha = 0.05
if p_value < alpha:
    print("拒绝原假设：高收入区房价显著更高")
else:
    print("无法拒绝原假设")

from scipy import stats

# 用 scipy 做简单线性回归（提供统计推断）
slope, intercept, r_value, p_value, std_err = stats.linregress(df['MedInc'], df['MedHouseVal'])

print(f"回归方程: MedHouseVal = {slope:.3f} * MedInc + {intercept:.3f}")
print(f"R² = {r_value ** 2:.3f}")
print(f"斜率 p 值: {p_value:.2e} → {'显著' if p_value < 0.05 else '不显著'}")
