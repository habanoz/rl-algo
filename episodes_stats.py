from numpy import ndarray


class EpisodesStats:
    def __init__(self, rewards: ndarray, cum_rewards: ndarray, lengths: ndarray, errors: ndarray):
        self.errors = errors
        self.lengths = lengths
        self.cum_rewards = cum_rewards
        self.rewards = rewards
