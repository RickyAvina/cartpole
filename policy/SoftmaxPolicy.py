import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, input_dim, output_dim, n_hidden, name):
		super(Actor, self).__init__()
		
		setattr(self, name+"_l1", nn.Linear(input_dim, n_hidden))
		setattr(self, name+"_l2", nn.Linear(n_hidden, output_dim))
		setattr(self, name+"_out_soft", nn.Softmax(dim=-1))

		self.name = name

	def forward(self, x):
		x = F.relu(getattr(self, self.name+"_l1")(x))
		x = F.relu(getattr(self, self.name+"_l2")(x))
		x = getattr(self, self.name+"_out_soft")(x)
		
		return x

class SoftmaxPolicy(object):
	def __init__(self, input_dim, output_dim, n_hidden, name="SoftmaxPolicy"):
		# set optimizer
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.actor = Actor(input_dim, output_dim, n_hidden, name)
		self.name = name
	
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1))
		action_probs = self.actor(state).detach().numpy()

		# sample action from action probability distribution
		action = np.random.choice(self.output_dim, p=action_probs.flatten())
		return action
