from abc import ABC, abstractmethod


class StateFlattener(ABC):
    @abstractmethod
    def flatten(self, obs):
        pass

    @abstractmethod
    def deflatten(self, state):
        pass

    @abstractmethod
    def deflatten_state_values(self, flat_state_values):
        pass

    def deflatten_action_values(self, action_values):
        pass

    def deflatten_policy(self, action_values):
        pass
