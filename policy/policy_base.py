from policy.SoftmaxPolicy import SoftmaxPolicy


class PolicyBase(object):
	def __init__(self, env):
		super(PolicyBase, self).__init__()
		
		self.env = env

	def set_dim(self):
		raise NotImplementedError()
	
	def select_stochastic_action(self):
		raise NotImplementedError()
	
	def set_policy(self):
		self.policy = SoftmaxPolicy(
			input_dim=self.input_dim,
			output_dim=self.output_dim,
			n_hidden=5	# generalize later
			)
	
	def save_weight(self, filename, directory):
		raise NotImplementedError()
	
	def load_weight(self, filename, directory):
		raise NotImplementedError()
