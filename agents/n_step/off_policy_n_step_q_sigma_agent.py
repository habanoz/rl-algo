from agents.base_agent import BaseAgent
import numpy as np

from model.agent_config import AgentConfig


class OffPolicyNStepQSigmaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentConfig, n_step_size=5, sigma=None, b_of_s=None,
                 b_of_a_given_s=None):
        super().__init__(config, f"OffPolicyNStepQSigmaAgent-n{n_step_size}")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_step_size = n_step_size

        self.Q = np.zeros((n_states, n_actions))
        if sigma is None:
            # self.sigma = np.ones(n_step_size + 1)  # full sampling
            # self.sigma = np.zeros(n_step_size + 1)  # pure expectation
            self.sigma = np.full(n_step_size + 1, 0.5)  # in half way
        else:
            self.sigma = np.array(sigma)

        assert len(self.sigma) == n_step_size + 1

        self.next_action = None
        self.t = None
        self.T = None
        self.rho = None

        self.observed_states = None
        self.selected_actions = None
        self.observed_rewards = None

        self.b_of_s = b_of_s
        if self.b_of_s is None:
            self.c.epsilon_decay = None  # epsilon should not decay, b is an exploratory policy
            self.b_of_s = lambda s: self.epsilon_greedy_action_select(self.Q[s, :])

        self.b_of_a_given_s = b_of_a_given_s
        if self.b_of_a_given_s is None:
            self.b_of_a_given_s = lambda a, s: ((1 - self.c.epsilon) + (self.c.epsilon / self.n_actions)) \
                if self.greedy_action_select(self.Q[s, :]) == a else (self.c.epsilon / self.n_actions)

        self.reset_episode_data()

    def reset_episode_data(self):
        self.next_action = None
        self.t = -1
        self.T = float("inf")
        self.rho = np.empty(self.n_step_size + 1, dtype=float)
        self.observed_states = np.empty(self.n_step_size + 1, dtype=int)
        self.selected_actions = np.empty(self.n_step_size + 1, dtype=int)
        self.observed_rewards = np.zeros(self.n_step_size + 1, dtype=int)

    def get_action(self, state):
        if self.next_action is not None:
            return self.next_action

        a0 = self.b_of_s(state)
        self.observed_states[0] = state
        self.selected_actions[0] = a0

        return a0

    def update(self, state, action, reward, done, next_state):
        self.t += 1

        if self.t < self.T:
            self.observed_rewards[self.modded(self.t + 1)] = reward
            self.observed_states[self.modded(self.t + 1)] = next_state

            self.next_action = self.b_of_s(next_state)
            self.selected_actions[self.modded(self.t + 1)] = self.next_action
            self.rho[self.modded(self.t + 1)] = \
                self.pi_a_s(self.next_action, next_state) / self.b_of_a_given_s(self.next_action, next_state)

            if done:
                self.T = self.t + 1

        tau = self.t - self.n_step_size + 1
        self.update_tau(tau)

        if done:
            for tau_p in range(tau + 1, self.T):
                self.t += 1
                self.update_tau(tau_p)

            self.reset_episode_data()

        super().update(state, action, reward, done, next_state)

    def update_tau(self, tau):

        if tau >= 0:
            G = 0
            for k in range(min(self.t + 1, self.T), tau, -1):  # tau + 1 + (-1) for a closed range
                if k == self.T:
                    G = self.observed_rewards[self.modded(self.T)]
                else:
                    s_k = self.observed_states[self.modded(k)]

                    v_bar = sum([
                        self.pi_a_s(a, s_k) *
                        self.Q[s_k, a]
                        for a in range(self.n_actions)
                    ])

                    a_k = self.selected_actions[self.modded(k)]

                    sigma_terms = (self.sigma[self.modded(k)] * self.rho[self.modded(k)] + (
                            1 - self.sigma[self.modded(k)]) * self.pi_a_s(a_k, s_k))

                    # print(G, sigma_terms, v_bar,"=>", s_k, a_k)

                    G = self.observed_rewards[self.modded(k)] + self.c.gamma * sigma_terms * (
                            G - self.Q[s_k, a_k]) + self.c.gamma * v_bar

                    # print(G)

            # add training error
            self.add_training_error(G, self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ])

            self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ] += self.c.alpha * (
                    G -
                    self.Q[
                        self.observed_states[self.modded(tau)],
                        self.selected_actions[self.modded(tau)]
                    ]
            )

    def recur_g(self, t, h):
        if t == self.T:
            return self.observed_rewards[self.modded(self.T)]
        elif t == h:
            s_t = self.observed_states[self.modded(t)]
            a_t = self.selected_actions[self.modded(t)]
            return self.Q[s_t, a_t]
        else:
            s_t = self.observed_states[self.modded(t)]

            v_bar = sum([
                self.pi_a_s(a, s_t) *
                self.Q[s_t, a]
                for a in range(self.n_actions)
            ])

            a_t = self.selected_actions[self.modded(t)]

            sigma_terms = (self.sigma[self.modded(t)] * self.rho[self.modded(t)] + (
                    1 - self.sigma[self.modded(t)]) * self.pi_a_s(a_t, s_t))

            return self.observed_rewards[self.modded(t)] + self.c.gamma * sigma_terms * (
                    self.recur_g(t + 1, h) - self.Q[s_t, a_t]) + self.c.gamma * v_bar

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def pi_a_s(self, a, s):
        return 0.9 if a in self.greedy_action_set(self.Q[s, :]) else 0.1

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])
