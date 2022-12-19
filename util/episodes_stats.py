from numpy import ndarray


class EpisodesStats:
    def __init__(self, rewards: ndarray, cum_rewards: ndarray, lengths: ndarray, value_errors: ndarray, training_errors):
        self.value_errors = value_errors
        self.training_errors = training_errors
        self.lengths = lengths
        self.cum_rewards = cum_rewards
        self.rewards = rewards
