import numpy as np

from agents.base_agent import BaseAgent
from model.agent_training_config import AgentTrainingConfig


class QLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "QLearningAgent")

        self.Q = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, done, next_state):
        next_q_value = 0 if done else max(self.Q[next_state, :])
        td_error = (reward + self.c.gamma * next_q_value - self.Q[state, action])

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, done, next_state)

    def action_values(self):
        return self.Q
