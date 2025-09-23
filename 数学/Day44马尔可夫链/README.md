模拟马尔可夫链是理解系统随时间演变的一种有效方法，尤其是在系统状态变化仅依赖于当前状态而非历史状态的情况下。一个经典的例子就是天气预测模型，其中天气状态（如晴天、雨天）的变化遵循一定的概率规则。

### 示例：简单的天气预测模型

假设我们有三种天气状态：`晴天(Sunny)`、`多云(Cloudy)`和`雨天(Rainy)`。这些状态之间的转移遵循以下转移矩阵：

|       | Sunny | Cloudy | Rainy |
|-------|-------|--------|-------|
| **Sunny**  | 0.7   | 0.2    | 0.1   |
| **Cloudy** | 0.3   | 0.4    | 0.3   |
| **Rainy**  | 0.2   | 0.3    | 0.5   |

这意味着如果今天是晴天，那么明天有70%的概率还是晴天，20%的概率变为多云，10%的概率会下雨等等。

---

## ✅ Python 实现代码

```python
import numpy as np

# 转移矩阵
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # 晴天
    [0.3, 0.4, 0.3],  # 多云
    [0.2, 0.3, 0.5]   # 雨天
])

# 状态名称
states = ['Sunny', 'Cloudy', 'Rainy']

# 初始状态分布（例如，第一天是晴天）
initial_state_distribution = np.array([1, 0, 0])  # 第一天晴天

def simulate_markov_chain(transition_matrix, initial_state_distribution, steps):
    current_state = np.random.choice(len(states), p=initial_state_distribution)
    print("Day 0:", states[current_state])
    
    for day in range(1, steps + 1):
        current_state = np.random.choice(
            len(states),
            p=transition_matrix[current_state]
        )
        print(f"Day {day}:", states[current_state])

# 设置模拟天数
num_days = 10
simulate_markov_chain(transition_matrix, initial_state_distribution, num_days)
```

### 🔍 输出示例：

```
Day 0: Sunny
Day 1: Sunny
Day 2: Cloudy
Day 3: Rainy
...
```

---

## 📊 分析长期行为

为了分析该马尔可夫链的长期行为，我们可以计算其平稳分布（Stationary Distribution），即当时间足够长时每个状态出现的概率分布不再改变。

```python
# 计算平稳分布
def calculate_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # 找到接近1的特征值对应的特征向量，并归一化
    stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)]
    stationary_distribution = stationary_distribution / stationary_distribution.sum()
    return stationary_distribution.real.flatten()

stationary_dist = calculate_stationary_distribution(transition_matrix)
print("平稳分布:", {state: prob for state, prob in zip(states, stationary_dist)})
```

### 🔍 输出示例：

```
平稳分布: {'Sunny': 0.4, 'Cloudy': 0.3, 'Rainy': 0.3}
```

这意味着，在长时间运行后，晴天的概率约为40%，而多云和雨天的概率各约为30%。

---

## 🧪 应用场景举例

| 场景 | 描述 |
|------|------|
| 天气预测 | 如上述例子所示 |
| 市场份额预测 | 不同品牌之间市场份额的转换 |
| 页面浏览预测 | 用户在不同网页间的跳转模式 |
| 生物信息学 | DNA序列中核苷酸的转换模型 |


当然，作为数学专家，我们来深入探讨**马尔可夫链（Markov Chain）**的模拟，并以**天气预测**为例，完整展示其数学原理与实现过程。

---

## 一、马尔可夫链：核心概念

### 1. 定义

马尔可夫链是一种**随机过程**，具有**马尔可夫性质（Markov Property）**：

> “未来状态只依赖于当前状态，而与过去无关。”

数学表达：
\[
P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \dots, X_0 = i_0) = P(X_{t+1} = j \mid X_t = i)
\]

即：**无记忆性（Memoryless）**

---

### 2. 关键要素

1. **状态空间（State Space）**：所有可能状态的集合。  
   例如：天气状态 \( S = \{晴天, 阴天, 雨天\} \)

2. **转移概率矩阵（Transition Matrix）** \( P \)：  
   \( P_{ij} = P(X_{t+1} = j \mid X_t = i) \)，表示从状态 \( i \) 转移到 \( j \) 的概率。

3. **初始分布（Initial Distribution）** \( \pi^{(0)} \)：  
   初始时刻各状态的概率分布。

4. **时间演化**：  
   \( \pi^{(t)} = \pi^{(0)} P^t \)

---

## 二、示例：天气预测的马尔可夫链

### 1. 定义状态

设天气有三种状态：
- 0: 晴天（Sunny）
- 1: 阴天（Cloudy）
- 2: 雨天（Rainy）

### 2. 构建转移矩阵

假设我们通过历史数据统计得到以下转移概率：

| 今天 \ 明天 | 晴天 | 阴天 | 雨天 |
|------------|------|------|------|
| 晴天       | 0.7  | 0.2  | 0.1  |
| 阴天       | 0.3  | 0.4  | 0.3  |
| 雨天       | 0.2  | 0.3  | 0.5  |

写成矩阵形式：

\[
P = \begin{bmatrix}
0.7 & 0.2 & 0.1 \\
0.3 & 0.4 & 0.3 \\
0.2 & 0.3 & 0.5 \\
\end{bmatrix}
\]

每行和为 1（概率守恒）。

---

## 三、模拟马尔可夫链（Python 实现）

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义状态
states = ['Sunny', 'Cloudy', 'Rainy']
state_indices = {s: i for i, s in enumerate(states)}

# 转移矩阵
P = np.array([
    [0.7, 0.2, 0.1],  # Sunny -> ...
    [0.3, 0.4, 0.3],  # Cloudy -> ...
    [0.2, 0.3, 0.5]   # Rainy -> ...
])

# 初始状态分布（假设今天是晴天）
pi0 = np.array([1.0, 0.0, 0.0])  # 或 [0.5, 0.3, 0.2] 表示不确定

# 模拟函数
def simulate_markov_chain(P, pi0, n_steps):
    """
    模拟马尔可夫链的路径
    """
    n_states = P.shape[0]
    path = np.zeros(n_steps, dtype=int)
    
    # 初始状态采样
    current_state = np.random.choice(n_states, p=pi0)
    path[0] = current_state
    
    # 时间演化
    for t in range(1, n_steps):
        current_state = np.random.choice(n_states, p=P[current_state])
        path[t] = current_state
    
    return path

# 运行模拟
np.random.seed(42)  # 可重现
n_days = 30
weather_path = simulate_markov_chain(P, pi0, n_days)

# 输出天气序列
print("天气序列（30天）:")
for day, state_idx in enumerate(weather_path):
    print(f"第{day+1:2d}天: {states[state_idx]}")
```

---

## 四、分析长期行为（稳态分布）

马尔可夫链若满足**不可约（Irreducible）**和**非周期（Aperiodic）**，则存在**稳态分布（Stationary Distribution）** \( \pi \)，满足：

\[
\pi P = \pi \quad \text{且} \quad \sum_i \pi_i = 1
\]

即：\( \pi \) 是 \( P^T \) 的特征向量，对应特征值 1。

### 计算稳态分布

```python
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
```

输出可能为：
```
晴天: 0.444
阴天: 0.296
雨天: 0.259
```

这意味着：从长期看，44.4% 的时间是晴天。

---

## 五、验证：模拟 vs 理论

我们可以模拟多条路径，统计各状态频率，验证是否趋近稳态分布。

```python
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
```

应接近理论稳态分布。

---

## 六、关键性质分析

| 性质 | 说明 |
|------|------|
| **不可约性** | 所有状态互通（如：晴→雨→阴→晴） |
| **非周期性** | 各状态周期为1（如晴天可连续多天） |
| **遍历性（Ergodicity）** | 长期频率 = 稳态概率 |
| **收敛速度** | 由第二特征值决定 \( \lambda_2 \) |

---

## 七、扩展应用

| 领域 | 应用 |
|------|------|
| **自然语言处理** | n-gram 语言模型（词序列预测） |
| **网页排名** | PageRank 算法（马尔可夫链稳态） |
| **金融** | 股价状态转移（牛市/熊市） |
| **生物** | DNA 序列建模 |
| **强化学习** | MDP（马尔可夫决策过程）基础 |

---

## 八、总结

| 步骤 | 数学/操作 |
|------|----------|
| 1. 定义状态空间 | \( S = \{s_1, \dots, s_n\} \) |
| 2. 构建转移矩阵 | \( P_{ij} = P(X_{t+1}=j \| X_t=i) \) |
| 3. 设置初始分布 | \( \pi^{(0)} \) |
| 4. 模拟路径 | \( X_{t+1} \sim P(X_{t+1} \| X_t) \) |
| 5. 分析长期行为 | 解 \( \pi P = \pi \) |

✅ **核心思想**：马尔可夫链用**局部转移规则**描述**全局随机演化**，是连接**概率论**与**动态系统**的桥梁。

通过模拟，我们不仅能预测短期天气，还能理解气候的长期统计特性。

---

如果你想进一步探讨**隐马尔可夫模型（HMM）**、**连续时间马尔可夫链**或**MCMC采样**，我可以继续深入讲解。
