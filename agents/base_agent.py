from abc import ABC
from copy import copy

import numpy as np

from agents.a_agent import AAgent
from model.agent_training_config import AgentTrainingConfig


class BaseAgent(AAgent, ABC):
    def __init__(self, config: AgentTrainingConfig, n_actions, n_states, identifier=None):
        self.c = copy(config)
        self.identifier = identifier
        self.n_actions = n_actions
        self.n_states = n_states

        self._incremental_training_error = 0
        self._n_incremental_training_errors = 0
        self.total_training_error = 0

        self.actions_to_take = None
        if config.actions_to_take is not None and len(config.actions_to_take) > 0:
            self.actions_to_take = list(config.actions_to_take)

    def get_desc(self):
        return self.identifier

    def get_action(self, obs):
        return self.epsilon_greedy_action_select(obs)

    def update(self, state, action, reward, done, next_state):

        if done:
            self.do_after_episode()

    def do_after_episode(self):
        if self.c.epsilon_decay is not None and self.c.epsilon_decay > 0:
            self.c.epsilon = max(self.c.epsilon - self.c.epsilon_decay, self.c.min_epsilon, 0.0)

        self.total_training_error = self._incremental_training_error / max(self._n_incremental_training_errors, 1)
        self._incremental_training_error = 0
        self._n_incremental_training_errors = 0

    def epsilon_greedy_action_select(self, state):
        # this is for debugging. Return one of the predefined actions, if defined...
        if self.actions_to_take is not None and len(self.actions_to_take) > 0:
            return self.actions_to_take.pop(0)

        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(self.n_actions)
        else:
            return self.greedy_action_select(state)

    def greedy_action_select(self, state):
        q_values = self.action_values()[state]
        return self.greedy_action_select_q_values(q_values)

    def greedy_action_select_q_values(self, q_values):
        max_val = np.max(q_values)
        return np.random.choice(np.where(q_values == max_val)[0])

    def greedy_action_set(self, state):
        q_values = self.action_values()[state]
        max_val = np.max(q_values)
        return np.where(q_values == max_val)[0]

    def add_training_error(self, error):
        self._incremental_training_error += abs(error)
        self._n_incremental_training_errors += 1

    def state_values_mean(self):
        return np.array([np.mean(r) for r in self.action_values()])

    def state_values_max(self):
        return np.array([np.max(r) for r in self.action_values()])

    def get_policy(self):
        return np.array(np.argmax(r) for r in self.action_values())
