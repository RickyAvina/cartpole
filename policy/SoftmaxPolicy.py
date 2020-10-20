import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden, name):
        super(Actor, self).__init__()

        setattr(self, name + "_l1", nn.Linear(input_dim, n_hidden))
        setattr(self, name + "_l2", nn.Linear(n_hidden, n_hidden))
        setattr(self, name + "_l3", nn.Linear(n_hidden, output_dim))

        self.name = name

    def forward(self, x):
        x = F.relu(getattr(self, self.name + "_l1")(x))
        x = F.relu(getattr(self, self.name + "_l2")(x))
        x = F.softmax(getattr(self, self.name + "_l3")(x), dim=1)

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
        action_probs = self.actor(state)

        # sample action from action probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # Return action and log probability
        return action.item(), m.log_prob(action)

    def get_policy_loss(self, log_probs, rewards, discount, eps=1e-8):
        # Get returns
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + discount * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Compute the REINFORCE loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        return torch.cat(policy_loss).sum()

    def train(self, replay_buffer, discount):
        debug = {"loss": 0}

        # Get k collected trajectories
        rewards, log_probs = replay_buffer.get_trajectories()

        policy_losses = []
        for reward, log_prob in zip(rewards, log_probs):
            policy_loss = self.get_policy_loss(log_probs, rewards, discount)
            policy_losses.append(policy_loss)

        policy_losses = torch.stack(policy_losses).sum()

        self.optimizer.zero_grad()
        policy_losses.backward()
        self.optimizer.step()

        debug["loss"] = policy_loss.cpu().data.item()
        return debug

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), directory + filename + ".pth")

    def load(self, filename, directory):
        path = directory + filename + ".pth"
        self.actor.load_state_dict(torch.load(path))
