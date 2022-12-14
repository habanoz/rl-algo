import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class SarsaAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "SarsaAgent")

        self.Q = np.zeros((n_states, n_actions))
        self.next_action = None

    def get_action(self, state):
        if self.next_action is not None:
            return self.next_action

        return self.epsilon_greedy_action_select(state)

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        self.next_action = None if terminated else self.epsilon_greedy_action_select(next_state)
        next_q_value = 0 if terminated else self.Q[next_state, self.next_action]

        td_error = (reward + self.c.gamma * next_q_value - self.Q[state, action])

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, terminated, next_state)

    def action_values(self):
        return self.Q
