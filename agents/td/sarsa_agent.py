from agents.base_agent import BaseAgent
import numpy as np


class SarsaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.9, alpha=0.1):
        super().__init__(epsilon=epsilon)
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.Q = np.zeros((n_states, n_actions))
        self.next_action = None

    def get_action(self, state):
        if self.next_action:
            return self.next_action

        return self.epsilon_greedy_action_select(self.Q[state, :])

    def update(self, state, action, reward, terminated, next_state):
        next_action = self.epsilon_greedy_action_select(self.Q[next_state, :])

        self.Q[next_state, action] += self.alpha * (
                reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

        self.next_action = next_action
