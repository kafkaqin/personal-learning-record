import numpy as np



transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

states = ['Sunny','Cloudy','Rainy']

initial_state_distribution =  np.array([1,0,0])


def calculate_stationary_distribution(transition_matrix):
    eigenvalues,eigenvectors=np.linalg.eig(transition_matrix.T)
    stationary_distribution=eigenvectors[:,np.isclose(eigenvalues,1)]
    stationary_distribution = stationary_distribution / stationary_distribution.sum()
    return stationary_distribution.real.flatten()
def simulate_markov_chain(transition_matrix,initial_state_distribution,steps):
    current_state = np.random.choice(len(states),p=initial_state_distribution)
    print("Day 0:",states[current_state])

    for day in range(1,steps+1):
        current_state = np.random.choice(len(states),
                                         p=transition_matrix[current_state])
        print(f"Day {day}:",states[current_state])


num_days = 10
simulate_markov_chain(transition_matrix,initial_state_distribution,num_days)


stationary_dist = calculate_stationary_distribution(transition_matrix)
print("平稳分布:",{state: prob for state,prob in zip(states,stationary_dist)})


import numpy as np
import matplotlib.pyplot as plt
states = ['Sunny','Cloudy','Rainy']
state_indices = {s:i for i,s in enumerate(states)}

p = np.array(
    [
        [0.7,0.2,0.1],
        [0.3,0.4,0.3],
        [0.2,0.3,0.5],
    ]
)

pi0 = np.array([1.0,0.0,0.0])
def simulate_markov_chain(P,pi0,n_steps):
    n_states = P.shape[0]
    path = np.zeros(n_states,dtype=int)
    current_state = np.random.choice(n_states,p=pi0)
    path[0] = current_state

    for t in range(1,n_steps):
        current_state = np.random.choice(n_states,p=P[current_state])
        path[t] = current_state
    return path
np.random.seed(42)
n_days = 30
weather_path = simulate_markov_chain(p,pi0,n_days)
for day,state_index in enumerate(weather_path):
    print(day,state_index)


def compute_stationary_distribution(P):
    """
    求解稳态分布 πP = π
    """
    n = P.shape[0]
    # 构造线性系统 (P^T - I)π = 0，加上 ∑π_i = 1
    A = P.T - np.eye(n)
    A = np.vstack([A[:-1], np.ones(n)])  # 前n-1行 + 概率和约束
    b = np.zeros(n)
    b[-1] = 1.0  # ∑π_i = 1

    pi = np.linalg.solve(A, b)
    return pi


# 计算
pi_stationary = compute_stationary_distribution(P)
print(f"\n稳态分布:")
for i, state in enumerate(states):
    print(f"{state}: {pi_stationary[i]:.3f}")

n_simulations = 1000
n_days = 100
final_states = np.zeros(3)

for _ in range(n_simulations):
    path = simulate_markov_chain(P, pi0, n_days)
    # 统计最后几天的状态（避免初始影响）
    final_states += np.bincount(path[-10:], minlength=3)

# 归一化
empirical_dist = final_states / (n_simulations * 10)
print(f"\n模拟频率（最后10天平均）:")
for i, state in enumerate(states):
    print(f"{state}: {empirical_dist[i]:.3f}")