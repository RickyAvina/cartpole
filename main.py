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
	agent = set_policy(env, args)

	# Start train
	train(agent=agent, env=env)


def set_policy(env, args):
	from policy.agent import Agent
	policy = Agent(env=env, args=args, name="SoftmaxAgent")

	return policy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	
	# Algorithm arguments
	parser.add_argument(
		"--tau", default=0.01, type=float,
		help="Target network update rate")
	parser.add_argument(
		"--batch-size", default=50, type=int,
		help="Batch size for both actor")
	parser.add_argument(
		"--ep_max_timesteps", default=100, type=int,
		help="Maximum timesteps for each episode")
	parser.add_argument(
		"--actor-lr", default=0.0001, type=float,
		help="Learning rate for actor")
	parser.add_argument(
		"--n-hidden", default=200, type=int,
		help="Number of hidden units")
	parser.add_argument(
		"--discount", default=0.99, type=float,
		help="Discount factor")

	args = parser.parse_args()
	args.log_name = "env_name: cartpole-v0"
	main(args=args)
