import numpy.random

from agents.base_agent import BaseAgent
import numpy as np


class DoubleQLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.01, gamma=0.9, alpha=0.1):
        super().__init__(epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha

        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def get_action(self, state):
        return self.epsilon_greedy_action_select(self.Q1[state, :] + self.Q2[state, :])

    def update(self, state, action, reward, done, next_state):
        if numpy.random.binomial(1, 0.5) == 1:
            self.Q1[state, action] += self.alpha * (
                        reward + self.gamma * self.Q1[next_state, self.greedy_action_select(self.Q2[next_state, :])] -
                        self.Q1[state, action])
        else:
            self.Q2[state, action] += self.alpha * (
                        reward + self.gamma * self.Q2[next_state, self.greedy_action_select(self.Q1[next_state, :])] -
                        self.Q2[state, action])

        super().update(state, action, reward, done, next_state)
