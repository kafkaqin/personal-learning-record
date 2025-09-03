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



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris,make_classification
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
print(f"原始数据形状:{X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"降维后数据的形状:{X_pca.shape}")
plt.figure(figsize=(10, 4))

# 原始数据前两个特征的投影（对比）
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("原始数据前两个特征")
plt.colorbar()

# PCA 降维后
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.xlabel("第一主成分 (PC1)")
plt.ylabel("第二主成分 (PC2)")
plt.title("PCA 降维后 (4D → 2D)")
plt.colorbar(scatter)

plt.tight_layout()
plt.savefig("pca.png")

print(f"每个主成分解释的方差比例:{pca.explained_variance_ratio_}")
print(f"累计解释方差比例:{np.cumsum(pca.explained_variance_ratio_)}")

total_var = sum(pca.explained_variance_ratio_)
print(f"前 2 个主成分保留了{total_var:.1%}的信息")

pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"要保留 95% 方差,需要:{pca_95.n_components_}个主成")

pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel("主成分数量")
plt.ylabel("累计解释方差比例")
plt.title("PCA 累计方差贡献率")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.legend()
plt.savefig("pca_95.png")


def manual_pca(X,n_components=2):
    cov_matrix = np.cov(X,rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]
    eig_vals = eig_vals[idx]
    components = eig_vecs[:, :n_components]
    X_pca = X @ components
    return X_pca,components,eig_vals
X_pca_manual,_,_=manual_pca(X_scaled,n_components=2)
print("Sklearn PCA和手动 PCA结果是否相近?",np.allclose(X_pca,X_pca_manual,atol=1e-10))


X_high,y_high=make_classification(
    n_samples=200,
    n_features=100,
    n_informative=20,
    n_redundant=30,
    n_clusters_per_class=1,
    random_state=42,
)

X_high_scaled = StandardScaler().fit_transform(X_high)
pca_high = PCA(n_components=2)
X_2d=pca_high.fit_transform(X_high_scaled)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_high, cmap='plasma', alpha=0.8)
plt.colorbar(label='类别')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("100维数据通过 PCA 降维到 2D 可视化")
plt.savefig("pca_high.png")