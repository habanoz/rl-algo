import numpy as np
from numpy import ndarray

from agents.base_agent import BaseAgent, AgentTrainingConfig
from util.tiles import IHT, tiles


class EpisodicSemiGradientNStepSarsaAgent(BaseAgent):
    def __init__(self, config: AgentTrainingConfig, n_actions, n_states, state_scale, num_of_tilings=8, n_step_size=5):
        super().__init__(config, n_actions, n_states, f"EpisodicSemiGradientNStepSarsaAgent n{n_step_size}")

        self.num_of_tilings = num_of_tilings
        self.c.alpha /= self.num_of_tilings

        self.iht = IHT(n_states)

        self.w = np.zeros(n_states)

        self.state_scale = state_scale
        self.n_step_size = n_step_size

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
        self.observed_states = np.empty(self.n_step_size + 1, dtype=ndarray)
        self.selected_actions = np.empty(self.n_step_size + 1, dtype=int)
        self.observed_rewards = np.empty(self.n_step_size + 1, dtype=int)

    def get_action(self, state):
        if self.next_action is not None:
            return self.next_action

        a0 = self.epsilon_greedy_action_select_q_values(state)

        self.observed_states[0] = state
        self.selected_actions[0] = a0

        return a0

    def update(self, state, action, reward, terminated, next_state, truncated=False):

        self.t += 1

        if self.t < self.T:
            self.observed_rewards[self.modded(self.t + 1)] = reward

            if not terminated:
                self.observed_states[self.modded(self.t + 1)] = next_state
                self.next_action = self.epsilon_greedy_action_select_q_values(next_state)
                self.selected_actions[self.modded(self.t + 1)] = self.next_action

            else:
                self.T = self.t + 1

                self.observed_states[self.modded(self.t + 1)] = None
                self.next_action = None
                self.selected_actions[self.modded(self.t + 1)] = -1

        tau = self.t - self.n_step_size + 1
        self.update_tau(tau)

        if terminated or truncated:
            if truncated and not terminated:
                self.T = tau + self.n_step_size + 1
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
                G += pow(self.c.gamma, self.n_step_size) * self.value_estimate(
                    self.observed_states[self.modded(tau + self.n_step_size)],
                    self.selected_actions[self.modded(tau + self.n_step_size)]
                )

            td_error = G - self.value_estimate(self.observed_states[self.modded(tau)],
                                               self.selected_actions[self.modded(tau)])

            # add training error
            self.add_training_error(td_error)

            self.w[
                self.x(self.observed_states[self.modded(tau)], self.selected_actions[self.modded(tau)])
            ] += self.c.alpha * td_error

    def epsilon_greedy_action_select_q_values(self, state):
        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(self.n_actions)
        else:
            q_estimates = np.array([self.value_estimate(state, a) for a in range(self.n_actions)])
            return self.greedy_action_select_q_values(q_estimates)

    def modded(self, idx):
        return idx % (self.n_step_size + 1)

    def value_estimate(self, s, a):
        assert s is not None
        assert a >= 0

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
