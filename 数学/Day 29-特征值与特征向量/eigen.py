import numpy as np
A =  np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]])
print("矩阵A:")
print(A)

eigenvalues,eigenvector=np.linalg.eig(A)
print("\n 特征值:",eigenvalues)
print("\n 特征向量:",eigenvector)

## Av=λv
lada1=eigenvalues[0]
v1 = eigenvector[:,0]
Av1 = A @ v1
ladav1 = lada1 * v1
print("\nAv1:",Av1)
print("\nladav1:",ladav1)
print("\n")
print(np.allclose(ladav1,Av1))

norm_v1=np.linalg.norm(v1)
print("\nnorm_v1:",norm_v1)

print("\n矩阵是否对称:?",np.allclose(A,A.T))
x,y=np.linalg.eigh(A)
print("\nx:",x)
print("\ny:",y)

value_only = np.linalg.eigvals(A)
print("\nvalue_only:",value_only)

X = np.array([[1,2],[3,4],[5,6],[7,8]])
X_centered = X - np.mean(X,axis=0)

C= np.cov(X_centered,rowvar=False)
print("\nC:",C)
value,vector=np.linalg.eig(C)
print("\nvalue:",value)
print("\nvector:",vector)
