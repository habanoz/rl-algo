from agents.base_agent import BaseAgent
import numpy as np

from model.agent_training_config import AgentTrainingConfig


class ExpectedSarsaAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "ExpectedSarsaAgent")

        self.Q = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, done, next_state):
        expected_next_q_value = 0 if done else sum(
            [self.pi_at_st(a, next_state) * self.Q[next_state, a]
             for a in range(self.n_actions)]
        )

        td_error = reward + self.c.gamma * expected_next_q_value - self.Q[state, action]

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        super().update(state, action, reward, done, next_state)

    def pi_at_st(self, at, st):
        return 1 if at == self.greedy_action_select(st) else 0

    def action_values(self):
        return self.Q
