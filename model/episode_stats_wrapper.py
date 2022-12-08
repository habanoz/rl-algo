import gymnasium as gym
import numpy as np


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_runs: int = 5, n_episodes: int = 1000):
        super().__init__(env)

        self.n_episodes = n_episodes
        self.rewards: np.ndarray = np.empty((n_runs, n_episodes))
        self.cum_rewards: np.ndarray = np.empty((n_runs, n_episodes))
        self.lengths: np.ndarray = np.empty((n_runs, n_episodes))
        self.episodes_cum_reward_sum = 0
        self.episode_index = -1

        self.episode_reward_sum = None
        self.episode_length = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        self.episode_reward_sum = 0
        self.episode_length = 0

        return obs, info

    def step(self, action):
        (
            obs,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action)

        self.episode_reward_sum += reward
        self.episode_length += 1

        if done or truncated:
            self.episode_index += 1
            run = self.episode_index // self.n_episodes
            episode = self.episode_index % self.n_episodes

            self.episodes_cum_reward_sum += self.episode_reward_sum

            self.rewards[run, episode] = self.episode_reward_sum
            self.cum_rewards[run, episode] = self.episodes_cum_reward_sum
            self.lengths[run, episode] = self.episode_length

            if episode == self.n_episodes - 1:
                self.episodes_cum_reward_sum = 0

        return (
            obs,
            reward,
            done,
            truncated,
            info,
        )
