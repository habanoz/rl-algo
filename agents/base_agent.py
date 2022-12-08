import numpy as np


class BaseAgent:
    def __init__(self, epsilon=0.5, epsilon_decay=None, min_epsilon=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._incremental_training_error = 0
        self.total_training_error = 0

    def get_action(self, obs):
        pass

    def update(self, state, action, reward, done, next_state):
        if done:
            self.do_after_episode()

    def do_after_episode(self):
        if self.epsilon_decay is not None and self.epsilon_decay > 0:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon, 0.0)

        self.total_training_error = self._incremental_training_error
        self._incremental_training_error = 0

    def greedy_action_select(self, nd_array1):
        max_val = np.max(nd_array1)
        return np.random.choice(np.where(nd_array1 == max_val)[0])

    def epsilon_greedy_action_select(self, nd_array1_q):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(len(nd_array1_q))
        else:
            return self.greedy_action_select(nd_array1_q)

    def add_training_error(self, new_estimate, old_estimate):
        self._incremental_training_error += new_estimate - old_estimate
