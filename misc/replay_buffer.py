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
		data: (state, next_state, action, reward, done)
		'''

		if len(self.storage) > 1e5:
			self.storage.pop(0)
		self.storage.append(data)
		
	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshpe(-1, 1)
