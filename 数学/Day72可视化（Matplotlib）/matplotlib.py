import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)

plt.figure(figsize=(8, 6))
plt.scatter(x, y,c='blue',label='数据点',alpha=0.7)
plt.title('散点图示例')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.legend()
plt.grid(True)
plt.savefig('test11.png')

colors = np.random.rand(100)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=colors,cmap='viridis', alpha=0.7)
plt.title('带颜色映射的散点图')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.legend()
plt.grid(True)
plt.savefig('test12.png')

data = np.random.rand(5, 5)
plt.figure(figsize=(6, 6))
heatmap = plt.imshow(data,cmap='hot', interpolation='nearest')
plt.title('热力图 (imshow)')
plt.colorbar(heatmap,label='数值大小')
plt.xticks(np.arange(5),['A','B','C','D','E'])
plt.yticks(np.arange(5),['1','2','3','4','5'])
plt.savefig('test13.png')

df = pd.DataFrame(data, columns=['A','B','C','D','E'],index=['1','2','3','4','5'])
plt.figure(figsize=(6, 6))
sns.heatmap(df,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Seaborn 热力图')
plt.savefig('test14.png')


from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

corr = df_iris.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True,fmt=".2f", cmap='coolwarm',square=True,
            cbar_kws={'shrink':.8})
plt.title("Iris 数据集特性相关性热力图")
plt.savefig('test15.png')