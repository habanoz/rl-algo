import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class OffPolicyNStepQSigmaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, n_step_size=5, sigma=None, b_of_s=None,
                 b_of_a_given_s=None, complete_recursion_with_q=False):
        super().__init__(config, n_actions, n_states, f"OffPolicyNStepQSigmaAgent-n{n_step_size}")
        self.n_step_size = n_step_size

        # in the book, the boxed algorithm omits the initial value of the G.
        # Outside the box, the recursion conditions are given.
        # Use of G_h:h = Q(S_h, A_h) is dictated.
        # However, this setup behaves differently from TB algorithm
        # if we consider that the Q(0) algorithm should behave like the TB algorithm.
        # So by default we initialize G as it is done in the TB algorithm and experiments show that
        # this way Q(0) behaves like TB.
        # Set complete_recursion_with_q para to true to enforce use of Q value.
        self.complete_recursion_with_q = complete_recursion_with_q

        self.Q = np.zeros((n_states, n_actions))
        if sigma is None:
            # self.sigma = np.ones(n_step_size + 1)  # full sampling
            self.sigma = np.zeros(n_step_size + 1)  # pure expectation
            # self.sigma = np.full(n_step_size + 1, 0.1)  # in half way
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
            self.b_of_s = lambda s: self.epsilon_greedy_action_select(s)

        self.b_of_a_given_s = b_of_a_given_s
        if self.b_of_a_given_s is None:
            self.b_of_a_given_s = lambda a, s: ((1 - self.c.epsilon) + (self.c.epsilon / self.n_actions)) \
                if self.greedy_action_select(s) == a else (self.c.epsilon / self.n_actions)

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

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        self.t += 1

        if self.t < self.T:
            self.observed_rewards[self.modded(self.t + 1)] = reward

            if not terminated:
                self.observed_states[self.modded(self.t + 1)] = next_state

                self.next_action = self.b_of_s(next_state)
                self.selected_actions[self.modded(self.t + 1)] = self.next_action
                self.rho[self.modded(self.t + 1)] = \
                    self.pi_a_s(self.next_action, next_state) / self.b_of_a_given_s(self.next_action, next_state)

            else:
                self.T = self.t + 1

                self.next_action = None
                self.selected_actions[self.modded(self.t + 1)] = -1
                self.rho[self.modded(self.t + 1)] = -1

        tau = self.t - self.n_step_size + 1
        self.update_tau(tau)

        if terminated:
            for tau_p in range(tau + 1, self.T):
                self.t += 1
                self.update_tau(tau_p)

            self.reset_episode_data()

        super().update(state, action, reward, terminated, next_state)

    def update_tau(self, tau):

        if tau >= 0:
            if self.t + 1 >= self.T:
                G = self.observed_rewards[self.modded(self.T)]
            else:
                s_t_p_1 = self.observed_states[self.modded(self.t + 1)]
                a_t_p_1 = self.selected_actions[self.modded(self.t + 1)]

                if self.complete_recursion_with_q:
                    G = self.Q[s_t_p_1, a_t_p_1]
                else:
                    G = self.observed_rewards[self.modded(self.t + 1)] + self.c.gamma * sum(
                        [
                            self.pi_a_s(a, s_t_p_1) * self.Q[s_t_p_1, a]
                            for a in range(self.n_actions)
                        ]
                    )

            for k in range(min(self.t, self.T - 1), tau, -1):  # tau + 1 + (-1) for a closed range

                s_k = self.observed_states[self.modded(k)]

                v_bar = sum([
                    self.pi_a_s(a, s_k) *
                    self.Q[s_k, a]
                    for a in range(self.n_actions)
                ])

                a_k = self.selected_actions[self.modded(k)]

                rho_k = self.rho[self.modded(k)]

                assert rho_k >= 0

                sigma_k = self.sigma[self.modded(k)]

                sigma_terms = (sigma_k * rho_k + (1 - sigma_k) * self.pi_a_s(a_k, s_k))

                G = self.observed_rewards[self.modded(k)] + self.c.gamma * sigma_terms * (
                        G - self.Q[s_k, a_k]) + self.c.gamma * v_bar

            td_error = (G - self.Q[self.observed_states[self.modded(tau)], self.selected_actions[self.modded(tau)]])

            # add training error
            self.add_training_error(td_error)

            self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ] += self.c.alpha * td_error

    def recur_g(self, k, h):
        if k == self.T:
            return self.observed_rewards[self.modded(self.T)]
        elif k == h:
            s_k = self.observed_states[self.modded(k)]
            a_k = self.selected_actions[self.modded(k)]
            return self.Q[s_k, a_k]
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

            return self.observed_rewards[self.modded(k)] + self.c.gamma * sigma_terms * (
                    self.recur_g(k + 1, h) - self.Q[s_k, a_k]) + self.c.gamma * v_bar

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def pi_a_s(self, a, s):
        return 1 if a == self.greedy_action_select(s) else 0

    def action_values(self):
        return self.Q
