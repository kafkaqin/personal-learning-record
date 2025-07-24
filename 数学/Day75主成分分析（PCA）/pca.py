import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

data = load_iris()
X = data.data
y = data.target

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

df = pd.DataFrame(data.principalComponents,
                  columns=['principal Component 1', 'principal Component 2'])

df['Target'] = y
plt.figure(figsize=(8,6))
sns.scatterplot(x='principal Component 1', y='principal Component 2',
                hue=df.Target.tolist(),palette=sns.color_palette("hsv", len(set(y))),
                data=df,legend='full')

plt.title('2 Component PCA')
plt.savefig('pca.png')

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))