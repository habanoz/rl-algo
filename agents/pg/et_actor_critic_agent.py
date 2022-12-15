import numpy as np
from numpy import ndarray

from agents.base_agent import BaseAgent, AgentTrainingConfig, Feature


class ETActorCriticAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, feature: Feature,
                 initial_theta: ndarray = None):
        super().__init__(config, n_actions, n_states, "ETActorCriticAgent")

        self.theta: ndarray = initial_theta
        if self.theta is None:
            self.theta = np.zeros(len(feature.s_a(0, 0)))

        self.w: ndarray = np.zeros(n_states)
        self.x = feature
        self.I = None
        self.z = None
        self.z_w = None

        self._reset()

    def _reset(self):
        self.I = 1

        self.z = np.zeros_like(self.theta)
        self.z_w = np.zeros_like(self.w)

    def get_action(self, state):
        return np.random.choice(range(self.n_actions), p=self.pi_s(state))

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        next_estimate = 0 if terminated else self.value_estimate(next_state)

        td_error = reward + self.c.gamma * next_estimate - self.value_estimate(state)
        self.add_training_error(td_error)

        self.z_w = (self.c.gamma * self.c.lambda_w) * self.z_w + self.value_estimate_gradient(state)
        self.z = (self.c.gamma * self.c.lambdaa) * self.z + self.I * self.grad_log_pi(action, state)

        self.w += (self.c.alpha_w * td_error) * self.z_w
        self.theta += (self.c.alpha * td_error) * self.z

        self.I *= self.c.gamma

        if terminated or truncated:
            self._reset()

        super().update(state, action, reward, terminated, next_state)

    def value_estimate(self, state):
        return self.w[state]

    def value_estimate_gradient(self, state):
        grad = np.zeros_like(self.w)
        grad[state] = 1.0
        return grad

    def grad_log_pi(self, a, s):
        # equation 13.9
        return self.x.s_a(s, a) - np.sum([self.pi_a_s(b, s) * self.x.s_a(s, b) for b in range(self.n_actions)],
                                         axis=0)

    def pi_a_s(self, a, s) -> float:
        # softmax function, equation 13.2
        # linear action preferences, equation 13.3

        # x.s matrix, theta vector, do(theta, x.s) requires transpose
        # dot(x.s, theta) does not require transpose
        exp_h = np.exp(np.dot(self.x.s(s), self.theta))

        pi = exp_h[a] / np.sum(exp_h)

        if np.isnan(pi):
            raise Exception("none encountered. Try lowering learning rate.")

        return pi

    def pi_s(self, s) -> ndarray:
        # softmax function, equation 13.2
        # linear action preferences, equation 13.3

        # x.s matrix, theta vector, do(theta, x.s) requires transpose
        # dot(x.s, theta) does not require transpose
        exp_h = np.exp(np.dot(self.x.s(s), self.theta))

        pi = exp_h / np.sum(exp_h)

        if np.any(np.isnan(pi)):
            raise Exception("none encountered. Try lowering learning rate.")

        return pi

    def action_values(self):
        pass
