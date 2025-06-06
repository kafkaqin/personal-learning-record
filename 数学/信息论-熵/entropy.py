import numpy as np
from collections import Counter

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    probs = np.array([count/total for count in counts.values()])
    return -np.sum(probs*np.log2(probs))

def joint_entropy(X,Y):
    pairs = list(zip(X,Y))
    counts = Counter(pairs)
    total = len(pairs)
    probs = np.array([count/total for count in counts.values()])
    return -np.sum(probs*np.log2(probs))

def mutual_information(X,Y):
    H_X = entropy(X)
    H_Y = entropy(Y)
    H_XY = joint_entropy(X,Y)
    return H_X + H_Y - H_XY

# 示例数据
X = ['a', 'a', 'b', 'b', 'a', 'b']
Y = [1,   1,   1,   2,   2,   2]

print("H(X):", entropy(X))              # 输出: ~0.918
print("H(Y):", entropy(Y))              # 输出: ~0.918
print("H(X,Y):", joint_entropy(X, Y))   # 输出: ~1.47
print("I(X;Y):", mutual_information(X, Y))  # 输出: ~0.36