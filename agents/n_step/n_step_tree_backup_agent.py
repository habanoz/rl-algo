from agents.base_agent import BaseAgent
import numpy as np


class NStepTreeBackupAgent(BaseAgent):
    def __init__(self, n_states, n_actions, n_step_size=5, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.01,
                 gamma=0.9, alpha=0.1):
        super().__init__(epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_step_size = n_step_size

        self.gamma = gamma
        self.alpha = alpha

        self.Q = np.zeros((n_states, n_actions))

        self.next_action = None
        self.t = None
        self.T = None

        self.observed_states = None
        self.selected_actions = None
        self.observed_rewards = None

        self.reset_episode_data()

    def reset_episode_data(self):
        self.next_action = None
        self.t = -1
        self.T = float("inf")
        self.observed_states = np.empty(self.n_step_size + 1, dtype=int)
        self.selected_actions = np.empty(self.n_step_size + 1, dtype=int)
        self.observed_rewards = np.empty(self.n_step_size + 1, dtype=int)

    def get_action(self, state):
        if self.next_action is not None:
            return self.next_action

        a0 = np.random.choice(self.n_actions)
        self.observed_states[0] = state
        self.selected_actions[0] = a0

        return a0

    def update(self, state, action, reward, done, next_state):
        self.t += 1

        if self.t < self.T:
            self.observed_rewards[self.modded(self.t + 1)] = reward
            self.observed_states[self.modded(self.t + 1)] = next_state

            self.next_action = np.random.choice(self.n_actions)
            self.selected_actions[self.modded(self.t + 1)] = self.next_action

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
            if self.t + 1 >= self.T:
                G = self.observed_rewards[self.modded(self.T)]
            else:
                G = self.observed_rewards[self.modded(self.t + 1)] + self.gamma * sum(
                    [
                        self.pi_a_st(a, self.t + 1) * self.Q[self.observed_states[self.modded(self.t + 1)], a]
                        for a in range(self.n_actions)
                    ]
                )

            for k in range(min(self.t, self.T - 1), tau, -1):  # tau + 1 + (-1) for a closed range
                ak = self.selected_actions[self.modded(k)]

                G = self.observed_rewards[self.modded(k)] + self.gamma * sum(
                    [
                        self.pi_a_st(a, k) * self.Q[self.observed_states[self.modded(k)], a]
                        for a in range(self.n_actions) if a != ak
                    ]
                ) + self.gamma * self.pi_a_st(ak, k) * G

            # add training error
            self.add_training_error(G, self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ])

            self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ] += self.alpha * (
                    G -
                    self.Q[
                        self.observed_states[self.modded(tau)],
                        self.selected_actions[self.modded(tau)]
                    ]
            )

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def pi_a_st(self, a, t):
        return 1 if a == self.greedy_action_select(
            self.Q[self.observed_states[self.modded(t)], :]) else 0
