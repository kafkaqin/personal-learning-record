import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.parse.transitionparser import Transition
import collections

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out_layer(x)
        return x

import random
from collections import deque
import math
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQNAgent:
    def __init__(self, state_space, action_space, device='cpu'):
        self.state_space = state_space
        self.action_space = action_space
        self.device = device

        self.policy_net = DQN(state_space.shape[0], action_space.n).to(device)
        self.target_net = DQN(state_space.shape[0], action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(100000)

        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).max(1)[1].view(1,1))
        else:
            return torch.tensor([[random.randrange(self.action_space.n)]], device=self.device,
                                dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < 100:
            return
        transitions = self.memory.sample(100)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),device=self.device,dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(100,device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * 0.99)+reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()