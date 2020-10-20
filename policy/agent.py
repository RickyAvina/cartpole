import numpy as np
from misc.replay_buffer import ReplayBuffer
from policy.policy_base import PolicyBase


class Agent(PolicyBase):
    def __init__(self, env, n_hidden, log, tb_writer, name, args):
        super(Agent, self).__init__(
            env=env, n_hidden=n_hidden, log=log, tb_writer=tb_writer,
            name=name, args=args)

        self.set_dim()
        self.set_policy()
        self.memory = ReplayBuffer()

    def set_dim(self):
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

        self.log[self.args.log_name].info("[{}] actor input dim: {}".format(
            self.name, self.input_dim))
        self.log[self.args.log_name].info("[{}] actor output dim: {}".format(
            self.name, self.output_dim))
        self.log[self.args.log_name].info("[{}] number of hidden neurons: {}"
                                          .format(self.name, self.n_hidden))

    def select_stochastic_action(self, obs):
        # Get probabilities for different actions
        action, log_prob = self.policy.select_action(obs)
        assert not np.isnan(action).any()
        return action, log_prob

    def clear_memory(self):
        self.memory.clear()

    def add_memory(self, reward, log_prob):
        self.memory.add(reward, log_prob)

    def update_policy(self, total_eps):
        debug = self.policy.train(
            replay_buffer=self.memory,
            discount=self.args.discount)

        self.log[self.args.log_name].info(
            "Training loss: {}".format(debug['loss']))
        self.tb_writer.add_scalar(
            "loss", debug['loss'], total_eps)
