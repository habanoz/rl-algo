import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.frozen_lake_plotter import FrozenLakePlotter
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from agents.td.double_q_learning_agent import DoubleQLearningAgent
from agents.td.q_learning_agent import QLearningAgent


def generate_episodes(env, agent, n_episodes=1000):
    for episode in tqdm(range(n_episodes)):

        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # update the agent
            agent.update(obs, action, reward, done, next_obs)

            obs = next_obs


if __name__ == '__main__':
    runs = 5
    n_episodes = 10_000
    demo_runs = False

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
                   render_mode=None)  # set render_mode to "Human"
    env = EpisodeStatsWrapper(env, n_runs=runs, n_episodes=n_episodes)

    for run in range(runs):
        agent = NStepSarsaAgent(env.observation_space.n, env.action_space.n, epsilon=1.0,
                                epsilon_decay=0.99 / n_episodes, n_step_size=2)

        generate_episodes(env, agent, n_episodes=n_episodes)

        if demo_runs and run == runs - 1:
            demo_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
            agent.epsilon = 0.0
            agent.epsilon_decay = None
            # generate_episodes(demo_env, agent, n_episodes=10)

            FrozenLakePlotter(agent.Q, 4, 4, "4x4 Slippery").show()

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
    axs[0].set_title("Episode rewards")
    rewards_means = np.mean(env.rewards, axis=0)

    axs[0].plot(range(n_episodes), rewards_means)

    axs[1].set_title("Episode lengths")
    length_means = np.mean(env.lengths, axis=0)
    axs[1].plot(range(n_episodes), length_means)

    plt.tight_layout()
    plt.show()
