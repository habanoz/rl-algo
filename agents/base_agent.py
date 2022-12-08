from abc import ABC, abstractmethod

import numpy as np

from model.agent_config import AgentConfig


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.c = config
        self._incremental_training_error = 0
        self.total_training_error = 0

        self.actions_to_take = None
        if config.actions_to_take is not None and len(config.actions_to_take) > 0:
            self.actions_to_take = list(config.actions_to_take)

    def get_action(self, obs):
        pass

    def update(self, state, action, reward, done, next_state):
        # print(f"state:{state} action:{action} reward:{reward} next_state:{next_state} done:{done}")

        if done:
            self.do_after_episode()

    def do_after_episode(self):
        if self.c.epsilon_decay is not None and self.c.epsilon_decay > 0:
            self.c.epsilon = max(self.c.epsilon - self.c.epsilon_decay, self.c.min_epsilon, 0.0)

        self.total_training_error = self._incremental_training_error
        self._incremental_training_error = 0

    def greedy_action_select(self, nd_array1):

        max_val = np.max(nd_array1)
        return np.random.choice(np.where(nd_array1 == max_val)[0])

    def epsilon_greedy_action_select(self, nd_array1_q):
        # this is for debugging. Return one of the predefined actions, if defined...
        if self.actions_to_take is not None and len(self.actions_to_take) > 0:
            return self.actions_to_take.pop(0)

        if np.random.binomial(1, self.c.epsilon) == 1:
            return np.random.choice(len(nd_array1_q))
        else:
            return self.greedy_action_select(nd_array1_q)

    def add_training_error(self, new_estimate, old_estimate):
        self._incremental_training_error += new_estimate - old_estimate

    @abstractmethod
    def state_values(self):
        raise Exception("Not implemented")
