from agents.a_agent import AAgent
from agents.wappers.state_coder import StateFlattener


class StateWrapperAgent(AAgent):
    def __init__(self, delegate: AAgent, flattener: StateFlattener):
        self.delegate = delegate
        self.flattener = flattener

    def get_action(self, obs):
        s = self.flattener.flatten(obs)
        return self.delegate.get_action(s)

    def update(self, state, action, reward, done, next_state):
        s = self.flattener.flatten(state)
        sp = self.flattener.flatten(next_state)

        return self.delegate.update(s, action, reward, done, sp)

    def state_values(self):
        states = self.delegate.state_values()
        return self.flattener.deflatten_state_values(states)

    def action_values(self):
        action_values = self.delegate.action_values()
        return self.flattener.deflatten_action_values(action_values)

    def get_policy(self):
        action_values = self.delegate.action_values()
        return self.flattener.deflatten_policy(action_values)

