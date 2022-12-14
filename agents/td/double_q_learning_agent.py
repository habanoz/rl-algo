import numpy as np
import numpy.random

from agents.base_agent import BaseAgent, AgentTrainingConfig


class DoubleQLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "DoubleQLearningAgent")

        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def get_action(self, state):
        return self.epsilon_greedy_action_select_merged(state)

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        if numpy.random.binomial(1, 0.5) == 1:
            next_q_value = 0 if terminated else self.Q2[next_state, self.greedy_action_select_q1(next_state)]
            td_error = reward + self.c.gamma * next_q_value - self.Q1[state, action]

            # add training error
            self.add_training_error(td_error)

            self.Q1[state, action] += self.c.alpha * td_error
        else:
            next_q_value = 0 if terminated else self.Q1[next_state, self.greedy_action_select_q2(next_state)]
            td_error = reward + self.c.gamma * next_q_value - self.Q2[state, action]

            # add training error
            self.add_training_error(td_error)

            self.Q2[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, terminated, next_state)

    def action_values(self):
        return (self.Q1 + self.Q2) / 2

    def greedy_action_select_q1(self, state):
        q_values = self.Q1[state]
        max_val = np.max(q_values)
        return np.random.choice(np.where(q_values == max_val)[0])

    def greedy_action_select_q2(self, state):
        q_values = self.Q2[state]
        max_val = np.max(q_values)
        return np.random.choice(np.where(q_values == max_val)[0])

    def epsilon_greedy_action_select_merged(self, state):
        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(self.n_actions)
        else:
            merged_q_values = self.Q1[state, :] + self.Q2[state, :]
            return self.greedy_action_select_q_values(merged_q_values)
