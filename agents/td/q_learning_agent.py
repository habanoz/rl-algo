import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class QLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "QLearningAgent")

        self.Q = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        next_q_value = 0 if terminated else max(self.Q[next_state, :])
        td_error = (reward + self.c.gamma * next_q_value - self.Q[state, action])

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, terminated, next_state)

    def action_values(self):
        return self.Q
