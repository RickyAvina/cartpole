from policy.SoftmaxPolicy import SoftmaxPolicy


class PolicyBase(object):
	def __init__(self, env, n_hidden, log, tb_writer, name, args=None):
		super(PolicyBase, self).__init__()
		
		self.env = env
		self.log = log
		self.n_hidden = n_hidden
		self.tb_writer = tb_writer
		self.name = name
		self.args = args

	def set_dim(self):
		raise NotImplementedError()
	
	def select_stochastic_action(self):
		raise NotImplementedError()
	
	def set_policy(self):
		self.policy = SoftmaxPolicy(
			input_dim=self.input_dim,
			output_dim=self.output_dim,
			n_hidden=self.n_hidden, 
			args=self.args
			)
	
	def save_weight(self, directory, filename):
		self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
		self.policy.save(directory=directory, filename=filename)

	def load_weight(self, directory, filename):
		self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
		self.policy.load(directory=directory, filename=filename)
