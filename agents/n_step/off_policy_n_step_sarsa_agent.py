from copy import copy

import numpy as np

from agents.base_agent import BaseAgent
from model.agent_training_config import AgentTrainingConfig


class OffPolicyNStepSarsaAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, n_step_size=5, b_of_s=None,
                 b_of_a_given_s=None):
        super().__init__(config, n_actions, n_states, f"OffPolicyNStepSarsaAgent n{n_step_size}")

        self.n_step_size = n_step_size

        self.Q = np.full((n_states, n_actions), 0.0)

        self.next_action = None
        self.t = None
        self.T = None
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

            if not done:
                self.observed_states[self.modded(self.t + 1)] = next_state
                self.next_action = self.b_of_s(next_state)
                self.selected_actions[self.modded(self.t + 1)] = self.next_action

            else:
                self.T = self.t + 1

                self.observed_states[self.modded(self.t + 1)] = -1
                self.next_action = None
                self.selected_actions[self.modded(self.t + 1)] = -1

        tau = self.t - self.n_step_size + 1
        self.update_tau(tau)

        if done:
            for tau_p in range(tau + 1, self.T):
                self.update_tau(tau_p)

            self.reset_episode_data()

        super().update(state, action, reward, done, next_state)

    def update_tau(self, tau):

        if tau >= 0:
            rho = np.prod([
                self.pi_ai_si(i) /
                self.b_of_a_given_s(self.selected_actions[self.modded(i)], self.observed_states[self.modded(i)])
                for i in range(tau + 1, min(tau + self.n_step_size - 1, self.T - 1) + 1)
            ])

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
            ] += self.c.alpha * rho * (
                td_error
            )

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def pi_ai_si(self, i):
        s_i = self.observed_states[self.modded(i)]
        a_i = self.selected_actions[self.modded(i)]

        return 1 if a_i == self.greedy_action_select(s_i) else 0

    def action_values(self):
        return self.Q
