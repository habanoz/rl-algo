import numpy as np

from agents.base_agent import BaseAgent
from model.agent_training_config import AgentTrainingConfig
from util.tiles import IHT, tiles


class EpisodicSemiGradientSarsaAgent(BaseAgent):
    def __init__(self, config: AgentTrainingConfig, n_actions, n_states, state_scale, num_of_tilings=8):
        super().__init__(config, n_actions, n_states, "EpisodicSemiGradientSarsaAgent")

        self.num_of_tilings = num_of_tilings
        self.c.alpha /= self.num_of_tilings

        self.next_action = None
        self.iht = IHT(n_states)

        self.w = np.zeros(n_states)

        self.state_scale = state_scale

    def get_action(self, obs):
        if self.next_action is not None:
            return self.next_action

        return self.epsilon_greedy_action_select_q_values(obs)

    def update(self, state, action, reward, done, next_state):
        if done:
            td_error = (reward - self.value_estimate(state, action))

            self.add_training_error(td_error)

            self.w[self.x(state, action)] += self.c.alpha * td_error

            self.next_action = None
        else:
            next_action = self.epsilon_greedy_action_select_q_values(next_state)

            q_estimate = self.value_estimate(state, action)
            next_q_estimate = self.value_estimate(next_state, next_action)
            td_error = (reward + self.c.gamma * next_q_estimate - q_estimate)

            self.w[self.x(state, action)] += self.c.alpha * td_error

            self.next_action = next_action

    def epsilon_greedy_action_select_q_values(self, state):
        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(self.n_actions)
        else:
            q_estimates = np.array([self.value_estimate(state, a) for a in range(self.n_actions)])
            return self.greedy_action_select_q_values(q_estimates)

    def value_estimate(self, s, a):
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
