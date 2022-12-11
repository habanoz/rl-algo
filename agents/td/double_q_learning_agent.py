import numpy.random

from agents.base_agent import BaseAgent
import numpy as np

from model.agent_config import AgentConfig


class DoubleQLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentConfig):
        super().__init__(config)
        self.n_states = n_states
        self.n_actions = n_actions

        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def get_action(self, state):
        return self.epsilon_greedy_action_select(self.Q1[state, :] + self.Q2[state, :])

    def update(self, state, action, reward, done, next_state):
        if numpy.random.binomial(1, 0.5) == 1:

            new_est = reward + self.c.gamma * self.Q2[next_state, self.greedy_action_select(self.Q1[next_state, :])]

            # add training error
            self.add_training_error(new_est, self.Q1[state, action])

            self.Q1[state, action] += self.c.alpha * (new_est - self.Q1[state, action])
        else:
            new_est = reward + self.c.gamma * self.Q1[next_state, self.greedy_action_select(self.Q2[next_state, :])]

            # add training error
            self.add_training_error(new_est, self.Q2[state, action])

            self.Q2[state, action] += self.c.alpha * (new_est - self.Q2[state, action])

        super().update(state, action, reward, done, next_state)

    def state_values(self):
        return np.array([np.mean((r1 + r2) / 2) for r1, r2 in zip(self.Q1, self.Q2)])

    def action_values(self):
        return (self.Q1 + self.Q2) / 2
