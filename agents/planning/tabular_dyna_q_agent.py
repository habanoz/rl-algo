import numpy as np

from agents.base_agent import BaseAgent
from model.agent_training_config import AgentTrainingConfig


class TabularDynaQAgent(BaseAgent):
    def __init__(self, n_obs, n_actions, config: AgentTrainingConfig, n_planning_steps=10):
        super().__init__(config)

        self.Q = np.zeros((n_obs, n_actions))
        self.model = {}

        self.n_planning_steps = n_planning_steps

    def get_action(self, state):
        return self.epsilon_greedy_action_select(self.Q[state, :])

    def update(self, state, action, reward, done, next_state):
        # add training error
        self.add_training_error(reward + self.c.gamma * max(self.Q[next_state, :]), self.Q[state, action])

        self.Q[state, action] += self.c.alpha * (
                reward + self.c.gamma * max(self.Q[next_state, :]) - self.Q[state, action])

        self.model[(state, action)] = (reward, next_state)

        self.do_planning()

        super().update(state, action, reward, done, next_state)

    def do_planning(self):
        observed_state_actions = np.array(list(self.model.keys()))

        for n in range(self.n_planning_steps):
            idx = np.random.choice(len(observed_state_actions))
            state, action = observed_state_actions[idx]
            reward, next_state = self.model[(state, action)]

            self.Q[state, action] += self.c.alpha * (
                    reward + self.c.gamma * max(self.Q[next_state, :]) - self.Q[state, action])

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])
