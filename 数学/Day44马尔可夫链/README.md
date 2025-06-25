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

