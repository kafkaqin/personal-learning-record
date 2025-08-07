import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])

print("向量 a:",a )
print("向量 b:",b )
print("-" * 30)

add = a + b
print("a + b =",add)

sub = a - b
print("a - b =",sub)

dot_product = np.dot(a,b)
print("a @ b  =",dot_product)

cross_product = np.cross(a,b)
print("a * b  =",cross_product)
norm_a = np.linalg.norm(a)
print("|a| =",norm_a)

unit_a = a / np.linalg.norm(a)
print("单位向量 a:",unit_a)

x = np.array([1,2])
y = np.array([3,4])
print("2D点积:",np.dot(x,y))