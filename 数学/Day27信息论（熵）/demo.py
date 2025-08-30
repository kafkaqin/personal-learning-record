import numpy as np
from collections import Counter
def entropy(probabilities):
    probabilities = [p for p in probabilities if p>0]
    return -sum(p*np.log2(p) for p in probabilities)

p_cion = [0.5,0.5]
H_coin = entropy(p_cion)
print(H_coin)

p_biased = [0.8,0.2]
H_biased = entropy(p_biased)
print(H_biased)

data = ['A','A','B','B','B','C','D']
freq = Counter(data)
total = len(data)
probs = [count/total for count in freq.values()]
h_data = entropy(probs)
print(h_data)

def join_entropy(joint_probs):
    return -sum(p*np.log2(p) for p in joint_probs if p >0)
joint_p =[0.3,0.2,0.1,0.4]
H_joint = join_entropy(joint_p)
print(H_joint)

def mutual_information(joint_probs,marginal_x,marginal_y):
    joint_probs = np.array(joint_probs)
    I = 0.0
    for i in range(joint_probs.shape[0]):
        for j in range(joint_probs.shape[1]):
            p_xy = joint_probs[i,j]
            if p_xy ==0:
                continue
            p_x = marginal_x[i]
            p_y = marginal_y[j]
            I+=p_xy * np.log2(p_xy/(p_x*p_y))
    return I

joint_p = [
    [0.3,0.2],
    [0.1,0.4]
]

marginal_x = [0.3+0.2,0.1+0.4]
marginal_y = [0.5+0.1,0.2+0.4]
MI = mutual_information(joint_p,marginal_x,marginal_y)
print(MI)
H_X = entropy(marginal_x)
H_Y = entropy(marginal_y)
H_XY = join_entropy([p for row in joint_p for p in row])
MI_from_entropy = H_X + H_Y - H_XY
print(MI_from_entropy)

from scipy.stats import entropy as scipy_entropy
from itertools import product


def estimate_mi_from_data(x, y):
    """
    从样本数据中估计互信息
    x, y: 等长的列表或数组
    """
    # 构建联合频率表
    joint_count = {}
    for xi, yi in zip(x, y):
        joint_count[(xi, yi)] = joint_count.get((xi, yi), 0) + 1

    total = len(x)
    joint_probs = np.array([count / total for count in joint_count.values()])

    # 边缘分布
    px = np.array([x.count(val) / total for val in set(x)])
    py = np.array([y.count(val) / total for val in set(y)])

    # 使用 scipy 计算熵（更稳定）
    H_X = scipy_entropy(px, base=2)
    H_Y = scipy_entropy(py, base=2)
    H_XY = scipy_entropy(joint_probs, base=2)

    return H_X + H_Y - H_XY


# 示例数据
x = ['晴', '晴', '雨', '雨', '晴', '雨']
y = ['高', '低', '低', '低', '低', '高']

mi_est = estimate_mi_from_data(x, y)
print(f"从数据估计的互信息: {mi_est:.4f} 比特")