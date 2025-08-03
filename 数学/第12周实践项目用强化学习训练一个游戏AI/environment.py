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