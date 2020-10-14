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
		
	def sample(self):
		# return reward arrays
		if len(self) == 0:
			print("EMPTY SAMPLE")
			return [], []
		# right now there is only one thing in here
		
		rewards, log_probs = [], []
		for sample in self.storage:
			rewards.append(sample[3])
			log_probs.append(sample[4])

		return rewards, log_probs 
		
