import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class NStepSarsaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, n_step_size=5):
        super().__init__(config, n_actions, n_states, f"NStepSarsaAgent n-{n_step_size}")

        self.n_step_size = n_step_size

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

        a0 = self.epsilon_greedy_action_select(state)
        self.observed_states[0] = state
        self.selected_actions[0] = a0

        return a0

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        self.t += 1

        if self.t < self.T:
            self.observed_rewards[self.modded(self.t + 1)] = reward

            if not terminated:
                self.observed_states[self.modded(self.t + 1)] = next_state
                self.next_action = self.epsilon_greedy_action_select(next_state)
                self.selected_actions[self.modded(self.t + 1)] = self.next_action

            else:
                self.T = self.t + 1

                self.observed_states[self.modded(self.t + 1)] = -1
                self.next_action = None
                self.selected_actions[self.modded(self.t + 1)] = -1

        tau = self.t - self.n_step_size + 1
        self.update_tau(tau)

        if terminated:
            for tau_p in range(tau + 1, self.T):
                self.update_tau(tau_p)

            self.reset_episode_data()

        super().update(state, action, reward, terminated, next_state)

    def update_tau(self, tau):

        if tau >= 0:
            G = sum([
                pow(self.c.gamma, i - tau - 1) * self.observed_rewards[self.modded(i)]
                for i in range(tau + 1, min(tau + self.n_step_size, self.T) + 1)
            ])

            if tau + self.n_step_size < self.T:
                G += pow(self.c.gamma, self.n_step_size) * self.Q[
                    self.observed_states[self.modded(tau + self.n_step_size)],
                    self.selected_actions[self.modded(tau + self.n_step_size)]
                ]

            td_error = G - self.Q[self.observed_states[self.modded(tau)], self.selected_actions[self.modded(tau)]]

            # add training error
            self.add_training_error(td_error)

            self.Q[
                self.observed_states[self.modded(tau)],
                self.selected_actions[self.modded(tau)]
            ] += self.c.alpha * td_error

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def action_values(self):
        return self.Q
