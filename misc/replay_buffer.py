class ReplayBuffer(object):
    def __init__(self):
        self.storage = []
    
    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0
    
    def add(self, data):
        # TODO Directly store reward and logprobs
        # So that you do not need to use for loop below
        self.storage.append(data)
        
    def sample(self):
        if len(self) == 0:
            raise ValueError("Should not be here")

        rewards, log_probs = [], []
        for sample in self.storage:
            rewards.append(sample[3])
            log_probs.append(sample[4])

        return rewards, log_probs 
