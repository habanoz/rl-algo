from agents.base_agent import BaseAgent
import numpy as np

from model.agent_config import AgentConfig


class QLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentConfig):
        super().__init__(config=config, identifier="QLearningAgent")
        self.n_states = n_states
        self.n_actions = n_actions

        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state):
        return self.epsilon_greedy_action_select(self.Q[state, :])

    def update(self, state, action, reward, done, next_state):
        # add training error
        self.add_training_error(reward + self.c.gamma * max(self.Q[next_state, :]), self.Q[state, action])

        self.Q[state, action] += self.c.alpha * (
                reward + self.c.gamma * max(self.Q[next_state, :]) - self.Q[state, action])

        super().update(state, action, reward, done, next_state)

    def state_values(self):
        return np.array([np.max(r) for r in self.Q])

    def action_values(self):
        return self.Q