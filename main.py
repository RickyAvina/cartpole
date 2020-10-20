import argparse
import os
from gym_env import make_env
from misc.utils import set_log
from trainer.train import train, test
from tensorboardX import SummaryWriter
import random
import numpy as np
import torch


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logging
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))

    # Create env
    env = make_env(args)

    # Set seeds 0 seed is odd
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize policy
    agent = set_policy(env, args.n_hidden, tb_writer, log, args)

    # load agent
    if args.mode == "test":
        agent.load_weight("pytorch_models/", args.test_model)
        test(agent=agent, env=env, log=log, tb_writer=tb_writer, args=args)
    else:
        train(agent=agent, env=env, log=log, tb_writer=tb_writer,
              num_samples=args.num_samples, args=args)


def set_policy(env, n_hidden, tb_writer, log, args):
    from policy.agent import Agent
    policy = Agent(env=env, n_hidden=n_hidden, tb_writer=tb_writer, log=log,
                   args=args, name="SoftmaxAgent")

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm arguments
    parser.add_argument(
        "--ep_max_timesteps", default=100, type=int,
        help="Maximum timesteps for each episode")
    parser.add_argument(
        "--lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--n_hidden", default=128, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--discount", default=0.99, type=float,
        help="Discount factor")
    parser.add_argument(
       "--num_samples", default=1, type=int,
       help="Number of samples to collect before\
       updating policy")

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Sets Gym, PyTorch, and Numpy seeds")
    parser.add_argument(
        "--mode", default="train", type=str,
        help="Choose between training and testing")
    parser.add_argument(
        "--test_model", default="", type=str,
        help="Specify model to test")

    args = parser.parse_args()
    args.log_name = "env::cartpole-v0-s_prefix::%s" % (args.prefix)

    main(args=args)
