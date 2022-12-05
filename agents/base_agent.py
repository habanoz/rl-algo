import numpy as np


class BaseAgent:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

        pass

    def get_action(self, obs):
        pass

    def update(self, state, action, reward, terminated, next_state):
        pass

    def greedy_action_select(self, nd_array1):
        max_val = np.max(nd_array1)
        return np.random.choice(np.where(nd_array1 == max_val)[0])

    def epsilon_greedy_action_select(self, nd_array1_q):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(len(nd_array1_q))
        else:
            return self.greedy_action_select(nd_array1_q)
