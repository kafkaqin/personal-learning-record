import numpy as np
A = np.array([[1,2,3],[4,5,6]])

B = np.array([[7,8,9],[10,11,12]])

print(A)
print(B)

print("-" *30)

print(A+B)
print("-" *30)
print(A-B)

print("-" *30)
D = np.dot(A,B.T)
print(D)

print("-" *30)
print(A.T)

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

Y = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

Z = np.dot(X, Y)
print("X × Y =")
print(Z)