import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig
from util.tiles import IHT, tiles


class DifferentialSemiGradientSarsaAgent(BaseAgent):
    def __init__(self, config: AgentTrainingConfig, n_actions, n_states, state_scale, num_of_tilings=8):
        super().__init__(config, n_actions, n_states, "DifferentialSemiGradientSarsaAgent")

        self.num_of_tilings = num_of_tilings
        self.c.alpha /= self.num_of_tilings

        self.next_action = None
        self.iht = IHT(n_states)

        self.w = np.zeros(n_states)
        self.r_bar = 0.0

        self.state_scale = state_scale

    def get_action(self, obs):
        if self.next_action is not None:
            return self.next_action

        return self.epsilon_greedy_action_select_q_values(obs)

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        self.next_action = None if terminated else self.epsilon_greedy_action_select_q_values(next_state)
        next_estimate = 0 if terminated else self.value_estimate(next_state, self.next_action)

        td_error = reward - self.r_bar + next_estimate - self.value_estimate(state, action)

        self.r_bar += self.c.beta * td_error
        self.w[self.x(state, action)] += self.c.alpha * td_error

    def epsilon_greedy_action_select_q_values(self, state):
        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(self.n_actions)
        else:
            q_estimates = np.array([self.value_estimate(state, a) for a in range(self.n_actions)])
            return self.greedy_action_select_q_values(q_estimates)

    def value_estimate(self, s, a):
        # return np.dot(self.w, self.x(s, a))
        return np.sum(self.w[self.x(s, a)])

    def x(self, state, action):
        return tiles(self.iht, self.num_of_tilings, state * self.state_scale, [action])

    def value_estimate_gradient(self, s, a):
        return self.x(s, a)

    def state_values_mean(self):
        pass

    def state_values_max(self):
        pass

    def action_values(self):
        pass

    def get_policy(self):
        pass
