import numpy as np
from misc.replay_buffer import ReplayBuffer
from policy.policy_base import PolicyBase


class Agent(PolicyBase):
	def __init__(self, env):
		super(Agent, self).__init__(env=env)
		
		self.set_dim()
		self.set_policy()
		self.memory = ReplayBuffer()
	
	def set_dim(self):
		self.input_dim = self.env.observation_space.shape[0]
		self.output_dim = self.env.action_space.n
		# do some logging here
	
	def select_deterministic_action(self, obs):
		action = self.policy.select_action(obs)
		assert not np.isnan(action).any()
		return action

	def select_stochastic_action(self, obs):
		# Get probabilities for different actions
		action = self.policy.select_action(obs)
		assert not np.isnan(action).any()
		return action

	def add_memory(self, obs, new_obs, action, reward, done):
		self.memory.add((obs, new_obs, action, reward, done))

	def update_policy(self):
		raise NotImplementedError()
