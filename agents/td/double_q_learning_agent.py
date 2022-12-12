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
            next_q_value = 0 if done else self.Q2[next_state, self.greedy_action_select(self.Q1[next_state, :])]
            td_error = reward + self.c.gamma * next_q_value - self.Q1[state, action]

            # add training error
            self.add_training_error(td_error)

            self.Q1[state, action] += self.c.alpha * td_error
        else:
            next_q_value = 0 if done else self.Q1[next_state, self.greedy_action_select(self.Q2[next_state, :])]
            td_error = reward + self.c.gamma * next_q_value - self.Q2[state, action]

            # add training error
            self.add_training_error(td_error)

            self.Q2[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, done, next_state)

    def state_values(self):
        return np.array([np.mean((r1 + r2) / 2) for r1, r2 in zip(self.Q1, self.Q2)])

    def action_values(self):
        return (self.Q1 + self.Q2) / 2
