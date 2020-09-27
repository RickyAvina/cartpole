import gym


def make_env(args):
	''' Create a gym environment given args '''
	
	env = gym.make('CartPole-v0')
	return env
