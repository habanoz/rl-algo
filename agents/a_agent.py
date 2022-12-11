from abc import ABC, abstractmethod


class AAgent(ABC):

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def update(self, state, action, reward, done, next_state):
        pass

    @abstractmethod
    def state_values(self):
        pass

    @abstractmethod
    def action_values(self):
        pass

    @abstractmethod
    def get_policy(self):
        pass
