import numpy as np
from numpy import ndarray, isnan

from agents.base_agent import BaseAgent, AgentTrainingConfig, Feature


class ReinforceSoftmaxLinearMcAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, feature: Feature,
                 initial_theta: ndarray = None):
        super().__init__(config, n_actions, n_states, "ReinforceSoftmaxLinearMcAgent")

        self.states = []
        self.actions = []
        self.rewards = [0]

        self.theta: ndarray = initial_theta
        if self.theta is None:
            self.theta = np.zeros(len(feature.s_a(0, 0)))

        self.x = feature

    def get_action(self, obs):
        pi = self.pi_s(obs)
        return np.random.choice(range(self.n_actions), p=pi)

    def update(self, state, action, reward, terminated, next_state, truncated=False):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if terminated or truncated:
            self.do_episode_ended()

        super().update(state, action, reward, terminated, next_state)

    def do_episode_ended(self):
        self.learn()
        self.do_after_episode()

        self.states = []
        self.actions = []
        self.rewards = [0]

    def learn(self):
        T = len(self.states)
        G = np.empty(T)

        G[-1] = self.rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = self.rewards[t + 1] + self.c.gamma * G[t + 1]

        for t in range(T):
            self.theta += (self.c.alpha * pow(self.c.gamma, t) * G[t]) * \
                          self.grad_log_pi(self.actions[t], self.states[t])

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
