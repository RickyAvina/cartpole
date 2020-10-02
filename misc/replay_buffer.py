""" Simple replay buffer to hold training samples """
import numpy as np


class ReplayBuffer(object):
	def __init__(self, buffer_size=1e5):
		self.storage = []
		self.buffer_size = buffer_size
	
	def __len__(self):
		return len(self.storage)

	def clear(self):
		self.storage.clear()
		assert len(self.storage) == 0
	
	def sync(self, memory):
		'''
		Set entries of current buffer to `memory's` buffer
		'''

		self.clear()
		for exp in memory.storage:
			self.storage.append(exp)
		
		assert len(memory) == len(self.storage)
	
	def add(self, data):
		'''
		data: (state, next_state, action, reward, log_prob, done)
		'''

		if len(self.storage) > 1e5:
			self.storage.pop(0)
		self.storage.append(data)
		
	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		states, next_states, actions, rewards, log_probs, dones = [], [], [], [], [], []

		for i in ind:
			state, next_state, action, reward, log_prob, done = self.storage[i]
			states.append(state)
			next_states.append(next_state)
			actions.append(action)
			rewards.append(reward)
			log_probs.append(log_prob)
			dones.append(done)

		return np.array(states), np.array(next_states), np.array(actions),\
			   np.array(rewards).reshape(-1, 1), np.array(log_probs).reshape(-1, 1),\
			   np.array(dones).reshape(-1, 1)
