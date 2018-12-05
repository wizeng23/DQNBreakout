#replay class
from collections import deque
import random

class Replay(object):
	def __init__(self, replay_length, batch_size):
		self.replay = deque(maxlen=replay_length):
		self.batch_size = batch_size

	def __len__(self):
		return len(self.replay)

	def add(self, state, action, reward, next_state, done):
		self.replay.append((state, action, reward, next_state, done))

	def sample(self):
		samples = random.sample(self.replay, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*samples)
		states = np.stack(states, axis=0)
		next_states = np.stack(next_states, axis=0)
		return states, actions, rewards, next_states, dones
