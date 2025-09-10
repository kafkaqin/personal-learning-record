import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler


faces = fetch_olivetti_faces(shuffle=True,random_state=42)
X = faces.data
y = faces.target
print(f"数据形状:{X.shape}")
print(f"人脸图像尺寸:{int(np.sqrt(X.shape[1]))}x{int(np.sqrt(X.shape[1]))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"降维后的形状:{X_pca.shape}")
print(f"前两个主成分解释方差比例:{pca.explained_variance_ratio_}")
print(f"累计解释方差:{sum(pca.explained_variance_ratio_):.1%}")

plt.figure(figsize=(12, 6))

# 创建颜色映射
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', alpha=0.8, s=60)
plt.colorbar(scatter, ticks=range(40), label="人物 ID")
plt.xlabel("第一主成分 (PC1)")
plt.ylabel("第二主成分 (PC2)")
plt.title("PCA 降维后的人脸数据可视化（Olivetti 数据集）")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# PCA 的主成分形状是 (n_components, n_features)
eigenfaces = pca.components_.reshape((2, 64, 64))  # 2 个主成分

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.title(f"特征脸 {i+1} (解释方差: {pca.explained_variance_ratio_[i]:.1%})")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 用前 50 个主成分重建
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)
X_reconstructed_scaled = pca_50.inverse_transform(X_pca_50)

# 注意：inverse_transform 返回的是标准化后的数据，需还原
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)

# 显示原始与重建图像对比
n_samples = 5
plt.figure(figsize=(12, 5))

for i in range(n_samples):
    # 原始图像
    plt.subplot(2, n_samples, i + 1)
    plt.imshow(X[i].reshape(64, 64), cmap='gray')
    plt.title("原始" if i == 0 else "", loc='left')
    plt.axis('off')

    # 重建图像
    plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(X_reconstructed[i].reshape(64, 64), cmap='gray')
    plt.title("重建" if i == 0 else "", loc='left')
    plt.axis('off')

plt.suptitle(f"用 50 个主成分重建人脸（保留 {sum(pca_50.explained_variance_ratio_):.1%} 信息）")
plt.tight_layout()
plt.show()

# 计算不同主成分数量的累计解释方差
pca_full = PCA().fit(X_scaled)

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), linewidth=2)
plt.xlabel("主成分数量")
plt.ylabel("累计解释方差比例")
plt.title("PCA 累计方差贡献率（人脸数据）")
plt.grid(True, alpha=0.3)

# 标记 95% 和 99%
for thr in [0.95, 0.99]:
    k = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= thr) + 1
    plt.axhline(y=thr, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=k, color='r', linestyle='--', alpha=0.7)
    plt.text(k+10, thr-0.03, f'{thr*100:.0f}% → {k} components', color='red')

plt.show()

# 打印结果
k_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
k_99 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.99) + 1
print(f"保留 95% 信息需 {k_95} 个主成分")
print(f"保留 99% 信息需 {k_99} 个主成分")