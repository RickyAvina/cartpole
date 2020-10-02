import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, input_dim, output_dim, n_hidden, name):
		super(Actor, self).__init__()
		
		setattr(self, name+"_l1", nn.Linear(input_dim, n_hidden))
		setattr(self, name+"_l2", nn.Linear(n_hidden, output_dim))

		self.name = name

	def forward(self, x):
		x = F.relu(getattr(self, self.name+"_l1")(x))
		x = F.relu(getattr(self, self.name+"_l2")(x))

		return x

class SoftmaxPolicy(object):
	def __init__(self, input_dim, output_dim, n_hidden, args, name="SoftmaxPolicy"):
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.actor = Actor(input_dim, output_dim, n_hidden, name)
		self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
		self.name = name
	
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1))

		# Run forward pass through nn
		nn_out = self.actor(state)

		# Get action probabilities
		action_probs = F.softmax(nn_out, dim=1)

		# sample action from action probability distribution
		m = Categorical(action_probs)
		action = m.sample()

		# return action and log probability
		return action.item(), m.log_prob(action).detach().numpy()[0]
	
	def train(self, replay_buffer, iterations, batch_size, discount, tau=None, policy_freq=None):
		debug = {"loss": 0}

		for it in range(iterations):
			# Sample replay buffer stochastically
			state, next_state, action, reward, log_prob, done = replay_buffer.sample(batch_size)
			states = torch.tensor(state, requires_grad=True)
			next_states = torch.tensor(next_state, requires_grad=True)
			actions = torch.tensor(action, requires_grad=False)
			rewards = torch.tensor(reward, requires_grad=True)
			log_probs = torch.tensor(log_prob, requires_grad=True)
			dones = torch.tensor(1-done, requires_grad=False)

			# find return
			discounted_rewards = self.get_discounted_rewards(rewards, discount)

			policy_loss = []

			# calculate loss
			for log_prob, R in zip(log_probs, discounted_rewards):
				policy_loss.append(-log_prob*R)

			self.optimizer.zero_grad()
			policy_loss = torch.cat(policy_loss).sum()
			policy_loss.backward()
			self.optimizer.step()

	def get_discounted_rewards(self, rewards, discount):
		discounted_rewards = []
		for t in range(len(rewards)):
			Gt = 0
			pw = 0
			for reward in rewards[t:]:
				Gt += discount**pw * reward
				pw += 1
			discounted_rewards.append(Gt)

		discounted_rewards = torch.tensor(discounted_rewards)

		# normalize and subtract from mean
		discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + np.finfo(np.float32).eps.item())
		return discounted_rewards
