import gym
import numpy as np
import random
from gym import wrappers

env = gym.make('FrozenLake-v1',is_slippery=True)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
episodes = 5000

for i in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done,_, info = env.step(action)

        q_table[state,action] = q_table[state,action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state])-q_table[state,action]
        )
        state = next_state
    epsilon = max(min_epsilon, epsilon*epsilon_decay)

print("Q-table 已训练完成")