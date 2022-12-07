import gymnasium as gym
import numpy as np
from gymnasium import spaces

START_STATE = 10


class RandomWalkEnv(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Discrete(21)
        self.start_state = START_STATE
        self._agent_location = None

        self.true_values = np.arange(-20, 22, 2) / 20.0
        self.true_values[0] = self.true_values[-1] = 0

        # We have 2 actions, corresponding to "left", "right"
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = START_STATE

        return self._agent_location, {}

    def step(self, action):
        # Map the action (element of {0,1}) to the direction we walk in
        direction = -1 if action == 0 else 1

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.observation_space.n - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == 0 or self._agent_location == 20
        reward = (-1 if self._agent_location == 0 else 1) if terminated else 0

        return self._agent_location, reward, terminated, False, {}

    def render(self):
        print("render not supported")
