from agents.base_agent import BaseAgent
import numpy as np

from model.agent_config import AgentConfig


class ExpectedSarsaAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentConfig):
        super().__init__(config=config, identifier="ExpectedSarsaAgent")
        self.n_states = n_states
        self.n_actions = n_actions

        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state):
        return self.epsilon_greedy_action_select(self.Q[state, :])

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
        return 1 if at == self.greedy_action_select(self.Q[st, :]) else 0

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])

    def action_values(self):
        return self.Q
