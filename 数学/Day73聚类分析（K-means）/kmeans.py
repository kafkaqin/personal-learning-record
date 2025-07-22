import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(42)
X,y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans,s=50,cmap='viridis')
centers = kmeans.cluster_centers_

plt.scatter(centers[:,0],centers[:,1], c='red',s=200,alpha=0.75,marker='x')
plt.title("K-means 聚类结果")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.savefig("kmeans.png")

sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11),sse)
plt.title("肘部法确定最优k值")
plt.xlabel('K 值')
plt.ylabel('SSE')
plt.savefig("sse.png")