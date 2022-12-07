import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.frozen_lake_plotter import FrozenLakePlotter
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from agents.n_step.n_step_tree_backup_agent import NStepTreeBackupAgent
from agents.n_step.off_policy_n_step_sarsa_agent import OffPolicyNStepSarsaAgent
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


def execute(agent_factory, env, n_runs=10, n_episodes=10_000, rolling_length=100):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)
    for run in range(n_runs):
        agent = agent_factory()

        generate_episodes(w_env, agent, n_episodes=n_episodes)

    rewards_means = np.mean(w_env.rewards, axis=0)
    reward_moving_average = (
            np.convolve(
                rewards_means, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    length_means = np.mean(w_env.lengths, axis=0)
    length_moving_average = (
            np.convolve(
                length_means, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    return reward_moving_average, length_moving_average


if __name__ == '__main__':
    runs = 10
    n_episodes = 10_000

    # set render_mode to "Human" to
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)
    n_obs = env.observation_space.n
    n_actions = env.action_space.n

    agent1 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.3,
                                     n_step_size=1)
    reward_moving_average, length_moving_average = execute(agent1, env, n_runs=runs, n_episodes=n_episodes)

    agent2 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.3,
                                     n_step_size=2)
    reward_moving_average2, length_moving_average2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes)

    agent3 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.3,
                                     n_step_size=3)
    reward_moving_average3, length_moving_average3 = execute(agent3, env, n_runs=runs, n_episodes=n_episodes)

    agent4 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.3,
                                     n_step_size=4)
    reward_moving_average4, length_moving_average4 = execute(agent4, env, n_runs=runs, n_episodes=n_episodes)

    agent5 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.3,
                                     n_step_size=5)
    reward_moving_average5, length_moving_average5 = execute(agent5, env, n_runs=runs, n_episodes=n_episodes)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
    axs[0].set_title("Episode rewards")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average, label="n-1")
    axs[0].plot(range(len(reward_moving_average2)), reward_moving_average2, label="n-2")
    axs[0].plot(range(len(reward_moving_average3)), reward_moving_average3, label="n-3")
    axs[0].plot(range(len(reward_moving_average4)), reward_moving_average4, label="n-4")
    axs[0].plot(range(len(reward_moving_average5)), reward_moving_average5, label="n-5")
    axs[0].legend()

    axs[1].set_title("Episode lengths")
    axs[1].plot(range(len(length_moving_average)), length_moving_average, label="n-1")
    axs[1].plot(range(len(length_moving_average2)), length_moving_average2, label="n-2")
    axs[1].plot(range(len(length_moving_average3)), length_moving_average3, label="n-3")
    axs[1].plot(range(len(length_moving_average4)), length_moving_average4, label="n-4")
    axs[1].plot(range(len(length_moving_average5)), length_moving_average5, label="n-5")
    axs[1].legend()

    os.system('spd-say "your program has finished"')

    plt.tight_layout()
    plt.show()
