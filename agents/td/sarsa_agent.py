import numpy as np

from agents.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.01, gamma=0.9, alpha=0.1):
        super().__init__(epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.n_states = n_states
        self.n_actions = n_actions

        self.gamma = gamma
        self.alpha = alpha

        self.Q = np.zeros((n_states, n_actions))
        self.next_action = None

    def get_action(self, state):
        if self.next_action:
            return self.next_action

        return self.epsilon_greedy_action_select(self.Q[state, :])

    def update(self, state, action, reward, done, next_state):
        next_action = self.epsilon_greedy_action_select(self.Q[next_state, :])

        # add training error
        self.add_training_error(reward + self.gamma * self.Q[next_state, next_action], self.Q[state, action])

        self.Q[state, action] += self.alpha * (
                reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

        self.next_action = None if done else next_action

        super().update(state, action, reward, done, next_state)
