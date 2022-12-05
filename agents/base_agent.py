import numpy as np


class BaseAgent:
    def __init__(self):
        pass

    def get_action(self, obs):
        pass

    def update(self, obs, action, reward, terminated, next_obs):
        pass

    def arg_max(self, nd_array1):
        max_val = np.max(nd_array1)
        return np.random.choice(np.where(nd_array1 == max_val)[0])
