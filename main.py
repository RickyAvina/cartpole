import argparse
import os
from gym_env import make_env
from trainer.train import train


def main(args):
	# Create directories
	if not os.path.exists("./logs"):
		os.makedirs("./logs")
	if not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")
	
	# Set logging
	
	# Create env
	env = make_env(args)

	# Set seeds
	env.seed(0)

	# Initialize policy
	agent = set_policy(env)

	# Start train
	train(agent=agent, env=env)


def set_policy(env):
	from policy.agent import Agent
	policy = Agent(env=env)

	return policy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	
	# add arguments
	
	args = parser.parse_args()
	args.log_name = "env_name: cartpole-v0"
	main(args=args)
