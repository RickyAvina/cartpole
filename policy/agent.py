import numpy as np
from misc.replay_buffer import ReplayBuffer
from policy.policy_base import PolicyBase


class Agent(PolicyBase):
	def __init__(self, env, name, args):
		super(Agent, self).__init__(env=env, name=name, args=args)
		
		self.set_dim()
		self.set_policy()
		self.memory = ReplayBuffer()
	
	def set_dim(self):
		self.input_dim = self.env.observation_space.shape[0]
		self.output_dim = self.env.action_space.n
		# do some logging here
	
	def select_deterministic_action(self, obs):
		action, log_prob = self.policy.select_action(obs)
		assert not np.isnan(action).any()
		return action

	def select_stochastic_action(self, obs):
		# Get probabilities for different actions
		action, log_prob = self.policy.select_action(obs)
		assert not np.isnan(action).any()
		return action, log_prob

	def add_memory(self, obs, new_obs, action, reward, done):
		self.memory.add((obs, new_obs, action, reward, done))

	def update_policy(self):
		if len(self.memory) > self.args.ep_max_timesteps:
			debug = self.policy.train(
				replay_buffer=self.memory,
				iterations=self.args.ep_max_timesteps,
				batch_size=self.args.batch_size,
				discount=self.args.discount,
				tau=self.args.tau)
