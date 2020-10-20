class ReplayBuffer(object):
    def __init__(self):
        self.rewards = []
        self.log_probs = []

    def __len__(self):
        assert len(self.log_probs) == len(self.rewards)
        return len(self.rewards)

    def clear(self):
        self.rewards.clear()
        self.log_probs.clear()
        assert len(self) == 0

    def add(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def get_trajectories(self):
        assert len(self) != 0
        return self.rewards, self.log_probs
