import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import default_rng

START_STATE = 3
rng = default_rng()


class AccessControlEnv(gym.Env):

    def __init__(self, time_limit=1000):
        self.time_limit = time_limit
        self.t = 0

        self.min_servers = 0
        self.max_servers = 10
        self.min_priority = 0
        self.max_priority = 3
        self.rewards = [1, 2, 4, 8]

        self.n_servers = None
        self.priority = None

        self.low = np.array([self.min_servers, self.min_priority], dtype=np.float32)
        self.high = np.array([self.max_servers, self.max_priority], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.t = 0

        self.n_servers = self.max_servers
        self.priority = rng.uniform(self.min_priority, self.max_priority + 1)

        return np.array([self.n_servers, self.priority]), {}

    def step(self, action):
        self.t += 1

        server_change = True if action == 1 and self.n_servers > 0 else False
        reward = 0

        if server_change:
            reward = self.rewards[self.priority]
            self.n_servers -= 1
            self.priority = rng.uniform(self.min_priority, self.max_priority + 1)

        # some servers may become available
        self.n_servers += sum(rng.binomial(1, 0.06, self.max_servers - self.n_servers))

        return np.array([self.n_servers, self.priority]), reward, self.t >= self.time_limit, False, {}

    def render(self):
        print("render not supported")
