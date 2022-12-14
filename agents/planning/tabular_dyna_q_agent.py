import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class TabularDynaQAgent(BaseAgent):
    def __init__(self, n_obs, n_actions, config: AgentTrainingConfig, n_planning_steps=10):
        super().__init__(config, n_actions, n_obs, f"TabularDynaQAgent n{n_planning_steps}")

        self.Q = np.zeros((n_obs, n_actions))
        self.model = {}

        self.n_planning_steps = n_planning_steps

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        next_q = 0 if terminated else max(self.Q[next_state, :])
        td_error = (reward + self.c.gamma * next_q - self.Q[state, action])

        # add training error
        self.add_training_error(td_error)

        self.Q[state, action] += self.c.alpha * td_error

        self.model[(state, action)] = (reward, next_state, terminated)

        self.do_planning()

        super().update(state, action, reward, terminated, next_state)

    def do_planning(self):
        observed_state_actions = np.array(list(self.model.keys()))

        for n in range(self.n_planning_steps):
            idx = np.random.choice(len(observed_state_actions))
            state, action = observed_state_actions[idx]
            reward, next_state, done = self.model[(state, action)]

            next_q = 0 if done else max(self.Q[next_state, :])
            self.Q[state, action] += self.c.alpha * (reward + self.c.gamma * next_q - self.Q[state, action])

    def action_values(self):
        return self.Q
