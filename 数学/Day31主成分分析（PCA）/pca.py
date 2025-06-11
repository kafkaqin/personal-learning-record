from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.dot(np.random.rand(2,2),np.random.randn(2,100)).T

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("原始的维度:",X.shape)
print("降维后的数据形状:",X_reduced.shape)
print("各组成分解释方差比例:",pca.explained_variance_ratio_)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1])
plt.title("Original Data")


plt.subplot(1,2,2)
plt.scatter(X_reduced[:,0],X_reduced[:,1])
plt.title("Reduced Data")

plt.tight_layout()
plt.savefig("pca.png")