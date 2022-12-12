from agents.base_agent import BaseAgent
import numpy as np

from model.agent_training_config import AgentTrainingConfig


class NStepTreeBackupAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, n_step_size=5, b_of_s=None):
        super().__init__(config, f"NStepTreeBackupAgent-n{n_step_size}")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_step_size = n_step_size


        self.Q = np.zeros((n_states, n_actions))

        self.next_action = None
        self.t = None
        self.T = None

        self.observed_states = None
        self.selected_actions = None
        self.observed_rewards = None

        self.b_of_s = b_of_s
        if self.b_of_s is None:
            self.c.epsilon_decay = None  # epsilon should not decay, b is an exploratory policy
            self.b_of_s = lambda s: self.epsilon_greedy_action_select(self.Q[s, :])

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
                G = self.observed_rewards[self.modded(self.t + 1)] + self.c.gamma * sum(
                    [
                        self.pi_a_st(a, self.t + 1) * self.Q[self.observed_states[self.modded(self.t + 1)], a]
                        for a in range(self.n_actions)
                    ]
                )

            for k in range(min(self.t, self.T - 1), tau, -1):  # tau + 1 + (-1) for a closed range
                ak = self.selected_actions[self.modded(k)]

                G = self.observed_rewards[self.modded(k)] + self.c.gamma * sum(
                    [
                        self.pi_a_st(a, k) * self.Q[self.observed_states[self.modded(k)], a]
                        for a in range(self.n_actions) if a != ak
                    ]
                ) + self.c.gamma * self.pi_a_st(ak, k) * G

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

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def pi_a_st(self, a, t):
        return 1 if a == greedy_action_select(
            self.Q[self.observed_states[self.modded(t)], :]) else 0

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])
