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
        # add training error
        self.add_training_error(reward + self.c.gamma * max(self.Q[next_state, :]), self.Q[state, action])

        self.Q[state, action] += self.c.alpha * (reward + self.c.gamma * sum(
            [self.pi_at_st(a, next_state) * self.Q[next_state, a] for a in range(self.n_actions)]
        ) - self.Q[state, action])

        super().update(state, action, reward, done, next_state)

    def pi_at_st(self, at, st):
        return 1 if at == self.greedy_action_select(self.Q[st, :]) else 0

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])
