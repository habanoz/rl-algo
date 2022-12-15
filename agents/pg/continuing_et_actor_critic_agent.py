import numpy as np
from numpy import ndarray

from agents.base_agent import BaseAgent, AgentTrainingConfig, StateFeature, ValueFeature


class ContinuingETActorCriticAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, feature: ValueFeature,
                 feature_w: StateFeature):
        super().__init__(config, n_actions, n_states, "ContinuingETActorCriticAgent")

        self.x: ValueFeature = feature
        self.x_w: StateFeature = feature_w

        self.theta = np.zeros(feature.dim())
        self.w: ndarray = np.zeros(feature_w.dim())

        self.r_bar = 0

        self.x = feature

        self.z = None
        self.z_w = None

        self._reset()

    def _reset(self):
        self.z = np.zeros_like(self.theta)
        self.z_w = np.zeros_like(self.w)

    def get_action(self, state):
        return np.random.choice(range(self.n_actions), p=self.pi_s(state))

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        next_estimate = 0 if terminated else self.value_estimate(next_state)

        td_error = reward - self.r_bar + next_estimate - self.value_estimate(state)
        self.r_bar += self.c.alpha_r * td_error
        self.add_training_error(td_error)

        self.z_w = self.c.lambda_w * self.z_w + self.value_estimate_gradient(state)
        self.z = self.c.lambdaa * self.z + self.grad_log_pi(action, state)

        self.w += (self.c.alpha_w * td_error) * self.z_w
        self.theta += (self.c.alpha * td_error) * self.z

        if terminated or truncated:
            self._reset()

        super().update(state, action, reward, terminated, next_state)

    def value_estimate(self, state):
        return np.sum(self.w[self.x_w.s(state)])

    def value_estimate_gradient(self, state):
        feature = np.zeros(self.x_w.dim())
        feature[self.x_w.s(state)] = 1

        return feature

    def grad_log_pi(self, a, s):
        # equation 13.9

        feature = np.zeros(self.x.dim())
        expected = np.zeros(self.x.dim())

        feature[self.x.s_a(s, a)] = 1

        for b in range(self.n_actions):
            expected[self.x.s_a(s, b)] += self.pi_a_s(b, s)

        return feature - expected

    def pi_a_s(self, a, s) -> float:
        # softmax function, equation 13.2
        # linear action preferences, equation 13.3

        prefs = np.array([np.sum(self.theta[self.x.s_a(s, b)]) for b in range(self.n_actions)])
        exp_h = np.exp(prefs)

        pi = exp_h[a] / np.sum(exp_h)

        if np.isnan(pi):
            raise Exception("none encountered. Try lowering learning rate.")

        return pi

    def pi_s(self, s) -> ndarray:
        # softmax function, equation 13.2
        # linear action preferences, equation 13.3

        prefs = np.array([np.sum(self.theta[self.x.s_a(s, b)]) for b in range(self.n_actions)])
        exp_h = np.exp(prefs)

        pi = exp_h / np.sum(exp_h)

        if np.any(np.isnan(pi)):
            raise Exception("none encountered. Try lowering learning rate.")

        return pi

    def action_values(self):
        pass
