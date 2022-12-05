import numpy.random

from agents.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def get_action(self, obs):
        return numpy.random.choice(self.actions)
