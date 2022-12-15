import gymnasium as gym
import numpy as np
from gymnasium import spaces

START_STATE = 3


class ShortCorridorEnv(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Discrete(4)
        self.start_state = 0
        self._agent_location = None

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

        if self._agent_location == 1:
            direction *= -1

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.observation_space.n - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == self.observation_space.n - 1
        reward = -1

        return self._agent_location, reward, terminated, False, {}

    def render(self):
        print("render not supported")
