#replay and schedule class
from collections import deque
import numpy as np
import random
import torch

class Replay(object):
    def __init__(self, replay_length, batch_size, cuda):
        self.replay = deque(maxlen=replay_length)
        self.batch_size = batch_size
        self.cuda = cuda

    def __len__(self):
        return len(self.replay)

    def add(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def sample(self):
        samples = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = [np.array(out) for out in zip(*samples)]
        return states, actions, rewards, next_states, dones

    def sample_tensor(self):
        samples = list(self.sample())
        samples[0] = samples[0].astype(np.float) / 255.0 # states now range (0.0, 1.0)
        samples[3] = samples[3].astype(np.float) / 255.0 # next_states
        samples[4] = samples[4] * 1.0
        tensor_samples = [torch.from_numpy(sample).type(torch.FloatTensor) for sample in samples]
        tensor_samples[1] = tensor_samples[1].type(torch.LongTensor)
        if self.cuda:
            tensor_samples = [sample.cuda() for sample in tensor_samples]
        return tensor_samples

class LinearSchedule(object):
    def __init__(self, start_eps, end_eps, steps):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.steps = steps
        self.count = 0

    def choose_random(self):
        self.count += 1
        if self.count <= self.steps:
            eps = self.start_eps - (self.start_eps - self.end_eps) * self.count / self.steps
        else:
            eps = self.end_eps
        return random.random() < eps

    def get_steps(self):
        return self.count
