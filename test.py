# run python -i test.py for testing stuff in shell
import torch
import numpy as np
import gym
from wrappers import make_atari, wrap_deepmind
from utils import LinearSchedule, Replay

env=wrap_deepmind(make_atari('BreakoutNoFrameskip-v4'))
state=env.reset()
state = np.array(state)
r = Replay(50, 3, False)
for i in range(100):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    r.add(state, action, reward, next_state, done)
    state = next_state
s, a, r, ns, d = r.sample_tensor()