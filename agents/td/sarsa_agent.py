import numpy as np

from agents.base_agent import BaseAgent
from model.agent_config import AgentConfig


class SarsaAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentConfig):
        super().__init__(config, "SarsaAgent")
        self.n_states = n_states
        self.n_actions = n_actions

        self.Q = np.zeros((n_states, n_actions))
        self.next_action = None

    def get_action(self, state):
        if self.next_action is not None:
            return self.next_action

        return self.epsilon_greedy_action_select(self.Q[state, :])

    def update(self, state, action, reward, done, next_state):
        self.next_action = None if done else self.epsilon_greedy_action_select(self.Q[next_state, :])
        next_q_value = 0 if done else self.Q[next_state, self.next_action]

        td_error = (reward + self.c.gamma * next_q_value - self.Q[state, action])

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, done, next_state)

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])

    def action_values(self):
        return self.Q
