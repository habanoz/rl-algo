import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.mc.off_policy_mc_agent import OffPolicyMcAgent
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
import numpy as np

from agents.td.sarsa_agent import SarsaAgent


def train_agent(env, agent, n_episodes=1000):
    for episode in tqdm(range(n_episodes)):

        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs


if __name__ == '__main__':
    runs = 10
    n_episodes = 1_000

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
                   render_mode=None)  # set render_mode to "Human"
    env = EpisodeStatsWrapper(env, n_runs=runs, n_episodes=n_episodes)
    for run in range(runs):
        agent = SarsaAgent(env.observation_space.n, env.action_space.n)

        train_agent(env, agent, n_episodes=n_episodes)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    rewards_means = np.mean(env.rewards, axis=0)

    axs[0].plot(range(n_episodes), rewards_means)

    axs[1].set_title("Episode lengths")
    length_means = np.mean(env.lengths, axis=0)
    axs[1].plot(range(n_episodes), length_means)

    plt.tight_layout()
    plt.show()
