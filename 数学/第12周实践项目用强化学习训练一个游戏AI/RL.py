from collections import deque

import pygame
import random
import numpy as np

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 30

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        self.reset()
    def reset(self):
        self.bird_y = 256
        self.bird_vel = 0
        self.bird_rect = pygame.Rect(60, self.bird_y, 34, 24)
        self.pipes = []
        self.score = 0
        self.done = False
        self._add_pipe()
        return self._get_state()
    def _add_pipe(self):
        gap = 100
        y = random.randint(100, 400 - gap)
        pipe = {
            'x' : SCREEN_WIDTH,
            'top': y - 320,
            'bottom': y + gap,
            'passed': False
        }
        self.pipes.append(pipe)
    def _get_state(self):
        if len(self.pipes) == 0:
            pipe_x = pipe_gap = 0
        else:
            pipe = self.pipes[0]
            pipe_x = pipe['x']
            pipe_gap = pipe['top'] + 320 + 50
        return np.array([
            self.bird_y/512.0,
        self.bird_vel/10.0,
        pipe_x/512.0,
        pipe_gap / 512.0], dtype=np.float32)

    def step(self, action):
        reward = 0.1
        self.bird_vel +=1
        self.bird_y += self.bird_vel
        self.bird_rect.y = self.bird_y

        for pipe in self.pipes:
            pipe['x'] -=2
            if pipe['x'] < 60 -52 and not pipe['passed']:
                pipe['passed'] = True
                self.score += 1
                reward = 1.0
        self.pipes = [p for p in self.pipes if p['x'] > -52]
        if self.pipes and self.pipes[-1]['x'] < SCREEN_WIDTH - 200:
            self._add_pipe()

        if self.bird_y > 512 or self.bird_y < 0:
            self.done = True
            reward = -1.0
        for pipe in self.pipes:
            top_rect = pygame.Rect(pipe['x'], -320,52,320)
            bottom_rect = pygame.Rect(pipe['x'] + pipe['bottom'], 52,320)
            if self.bird_rect.colliderect(top_rect) or self.bird_rect.colliderect(bottom_rect):
                self.done = True
                reward = -1.0
        next_state = self._get_state()
        return next_state, reward, self.done,{}

    def render(self):
        self.screen.fill((135, 206, 250))
        pygame.draw.rect(self.screen, (255 ,255, 0), self.bird_rect)
        for pipe in self.pipes:
            pygame.draw.rect(self.screen,(0,128,0),(pipe['x'],-320,52,320))
            pygame.draw.rect(self.screen,(0,128,0),(pipe['x'],pipe['bottom'],52,320))
        pygame.display.update()
        self.clock.tick(FPS)


################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self,input_dim,output_dim=2):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,output_dim)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


##################
class DQNAgent():
    def __init__(self,state_dim,action_dim,lr=1e-4,gamma=0.99,epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=50000)
        self.batch_size = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim,action_dim).to(self.device)
        self.target_network = DQN(state_dim,action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_netwotk()

    def update_target_netwotk(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.vstack([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array(e[1] for e in batch)).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (~dones)
        target_q_values = target_q_values.unsqueeze(1)

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


import pygame
import sys
import random

pygame.init()
bird_movement = 0
score = 0

def create_pipe():
    height = random.randint(150,400)
    bottom_pipe = pygame.Rect(400,height,60,500)
    top_pipe = pygame.Rect(400,height-700,60,500)
    return bottom_pipe, top_pipe
def move_pipe(pipes):
    for pipe in pipes:
        pipe.centerx -= 3
    return [pipe for pipe in pipes if pipe.right > -50]

def get_state(bird_y,bird_velocity,pipes):
    if not pipes:
        return np.array([bird_y/700,bird_velocity/10,10,10])
    nearest_pipe = min([p for p in pipes if p.centerx > 50],key=lambda p: p.centerx,default=None)
    if nearest_pipe is None:
        dx = 400
        dy = 0
    else:
        dx = (nearest_pipe.centerx - 50) / 400
        dy = (nearest_pipe.top - bird_y) / 700
    return np.array([bird_y/700,bird_velocity/10,dx,dy])

def main():
    agent = DQNAgent(state_dim=4,action_dim=2)
    episodes = 1000
    best_score = 0

    for episode in range(episodes):
        bird_rect = pygame.Rect(50, 350, 30, 30)
        pipes = []
        frame_count = 0
        score = 0
        done = False
        bird_movement = 0

        print(f"Episode {episode + 1}/{episodes} , Epsilon: {agent.epsilon:3.f}")
        while not done:
            reward = 0.1
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state = get_state(bird_rect.centery,bird_movement,pipes)
            if np.random.rand() > 0.1:
                action = agent.act(state)
            else:
                action = random.randrange(2)

            if action == 1:
                bird_movement += -6

            bird_movement += gravity
            bird_rect.centery += bird_movement

            if frame_count % 150 == 0:
                b,t = create_pipe()
                pipes.extend([b,t])

            pipes = move_pipe(pipes)
            frame_count += 1

            if check_collision(pipes,bird_rect):
                done = True
                reward = -10
            else:
                for pipe in pipes:
                    if pipe.centerx == 50 and pipe.bottom > 600:
                        score += 1
                        reward = 10

            next_state = get_state(bird_rect.centery,bird_movement,pipes)
            agent.remember(state, action, reward, next_state, done)

            agent.replay()

            screen.fill((135,206,250))
            pygame.draw.rect(screen,(255,0,0),bird_rect)
            for pipe in pipes:
                pygame.draw.rect(screen,(0,128,0),pipe)
            pygame.display.flip()
            clock.tick(60)
            if done:
                print(f"Episode {episode + 1}/{episodes} , Score: {score}, Best: {best_score}")
                if score > best_score:
                    best_score = score
                    torch.save(agent.q_network.state_dict(),'model.pth')
                    break

        if episode % 10 == 0:
            agent.update_target_netwotk()

    pygame.quit()

if __name__ == '__main__':
    main()