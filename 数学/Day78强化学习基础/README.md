下面是一个使用 **OpenAI Gym** 和 **Q-learning** 算法解决简单迷宫问题的完整示例。我们将使用 Gym 中的 `FrozenLake-v1` 环境，它是一个经典的迷宫（grid world）问题。

---

## 🧩 问题描述：FrozenLake-v1

- 环境是一个 4x4 的网格迷宫。
- 玩家从起点 S（(0,0)）出发，目标是安全地走到终点 G（(3,3)）。
- 每个格子可能是：
    - `S`: Start（起点）
    - `F`: Frozen（安全）
    - `H`: Hole（掉下去就失败）
    - `G`: Goal（终点）
- 动作空间：`0=左`, `1=下`, `2=右`, `3=上`
- 状态空间：共 16 个状态（0~15）

---

## 🧠 Q-learning 简介

Q-learning 是一种无模型（model-free）的强化学习算法，用于学习一个 Q 表（Q-table）：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'}Q(s', a') - Q(s, a) \right]
$$

其中：
- $ s $: 当前状态
- $ a $: 当前动作
- $ r $: 奖励
- $ s' $: 下一状态
- $ \alpha $: 学习率（learning rate）
- $ \gamma $: 折扣因子（discount factor）

---

## 🧪 步骤概览

1. 创建环境
2. 初始化 Q-table
3. 使用 ε-greedy 策略选择动作
4. 更新 Q-table
5. 训练模型
6. 测试策略

---

## 🧰 安装 OpenAI Gym

```bash
pip install gym
```

---

## 📊 示例代码：Q-learning 解决 FrozenLake

```python
import gym
import numpy as np
import random
from gym import wrappers

# 创建环境
env = gym.make('FrozenLake-v1', is_slippery=True)  # 可以设置 is_slippery=False 来简化问题

# 初始化 Q-table（16 个状态，4 个动作）
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 超参数
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
episodes = 5000

# Q-learning 训练
for i in range(episodes):
    state = env.reset()[0]  # 返回的是一个 tuple，取第一个元素
    done = False

    while not done:
        # ε-greedy 策略选择动作
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机动作
        else:
            action = np.argmax(q_table[state])

        # 执行动作
        next_state, reward, done, _, info = env.step(action)

        # 更新 Q-table
        q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state

    # 衰减 epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Q-table 已训练完成！")
```

---

## 🧪 测试训练好的 Q-learning 策略

```python
# 测试策略
state = env.reset()[0]
done = False
steps = 0

print("测试路径：")
while not done and steps < 100:
    action = np.argmax(q_table[state])
    print(f"状态 {state} → 动作 {action}")
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    steps += 1

if reward == 1:
    print("🎉 成功到达终点！")
else:
    print("💀 掉入陷阱或超时。")
```

---

## 📈 可视化 Q-table（可选）

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制 Q-table
plt.figure(figsize=(10, 6))
sns.heatmap(q_table, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Q-table (状态 x 动作)")
plt.xlabel("动作")
plt.ylabel("状态")
plt.show()
```

---

## 📋 输出示例

```
Q-table 已训练完成！

测试路径：
状态 0 → 动作 1
状态 4 → 动作 1
状态 8 → 动作 2
状态 9 → 动作 1
状态 13 → 动作 2
状态 14 → 动作 2
🎉 成功到达终点！
```

---

## 🧠 小贴士

| 项目 | 说明 |
|------|------|
| `is_slippery=True` | 环境具有不确定性，增加难度 |
| `epsilon_decay` | 控制探索与利用的平衡 |
| `learning_rate` | 学习步长，不能太大也不能太小 |
| `discount_factor` | 未来奖励的折扣，越接近 1 越重视长期收益 |

---

## ✅ 扩展建议

- 使用神经网络替代 Q-table（即 DQN）
- 使用 `gym.wrappers.Monitor` 录制训练过程
- 更复杂的迷宫（如 8x8）
- 添加可视化界面（如 PyGame）

---