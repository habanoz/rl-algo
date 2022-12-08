import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from agents.n_step.off_policy_n_step_q_sigma_agent import OffPolicyNStepQSigmaAgent
from agents.td.sarsa_agent import SarsaAgent
from episodes_stats import EpisodesStats
from rms_errors_for_baseline import RMSErrorsForBaseline
from util.serialize_helper import serialize_values, deserialize_values


def generate_baseline(env, name: str, n_episodes=1000):
    agent = OnPolicyFirstVisitMcAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes,
                                      min_epsilon=0.01)

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

    v = np.array([np.mean(r) for r in agent.Q])
    serialize_values(v, name)


def generate_episodes(env, agent, n_episodes=1000, value_baseline=None):
    errors = np.empty(n_episodes)
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

        state_values = np.array([np.mean(r) for r in agent.Q])
        errors[episode] = RMSErrorsForBaseline(value_baseline).calculate_for_state(state_values)

    return errors


def execute(agent_factory, env, n_runs=10, n_episodes=10_000, value_baseline=None):
    rolling_length = n_episodes // 100

    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    rms_errors = np.zeros(n_episodes)
    for run in range(n_runs):
        agent = agent_factory()

        rms_errors += generate_episodes(w_env, agent, n_episodes=n_episodes, value_baseline=value_baseline)

    rms_errors = rms_errors / runs

    rewards_means = np.mean(w_env.rewards, axis=0)
    reward_moving_average = (
            np.convolve(
                rewards_means, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    cum_rewards_means = np.mean(w_env.cum_rewards, axis=0)
    cum_reward_moving_average = (
            np.convolve(
                cum_rewards_means, np.ones(rolling_length), mode="valid"
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

    rms_error_moving_average = (
            np.convolve(
                rms_errors, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    return EpisodesStats(reward_moving_average, cum_reward_moving_average, length_moving_average,
                         rms_error_moving_average)


def train_4(n_obs: int, n_actions: int, runs: int, n_episodes: int, value_baseline: ndarray = None):
    label1 = "sarsa"
    label2 = "sarsa n-1"
    label3 = "q-sigma n-2"
    label4 = "MC"

    # agent 1
    agent1 = lambda: SarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes,
                                min_epsilon=0.01, alpha=0.05)
    stats1 = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 2
    agent2 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.05,
                                     n_step_size=1)
    stats2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 3
    agent3 = lambda: OffPolicyNStepQSigmaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes, alpha=0.05,
                                     n_step_size=2)
    stats3 = execute(agent3, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 4
    agent4 = lambda: OnPolicyFirstVisitMcAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=0.99 / n_episodes)
    stats4 = execute(agent4, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(20, 10))

    x_ticks = range(len(stats1.rewards))

    axs[0].set_title("Episode rewards")
    axs[0].plot(x_ticks, stats1.rewards, label=label1)
    axs[0].plot(x_ticks, stats2.rewards, label=label2)
    axs[0].plot(x_ticks, stats3.rewards, label=label3)
    axs[0].plot(x_ticks, stats4.rewards, label=label4)
    axs[0].legend()

    axs[1].set_title("Cumulative Rewards")
    axs[1].plot(x_ticks, stats1.cum_rewards, label=label1)
    axs[1].plot(x_ticks, stats2.cum_rewards, label=label2)
    axs[1].plot(x_ticks, stats3.cum_rewards, label=label3)
    axs[1].plot(x_ticks, stats4.cum_rewards, label=label4)
    axs[1].legend()

    axs[2].set_title("Episode Lengths")
    axs[2].plot(x_ticks, stats1.lengths, label=label1)
    axs[2].plot(x_ticks, stats2.lengths, label=label2)
    axs[2].plot(x_ticks, stats3.lengths, label=label3)
    axs[2].plot(x_ticks, stats4.lengths, label=label4)
    axs[2].legend()

    axs[3].set_title("RMS Errors")
    axs[3].plot(x_ticks, stats1.errors, label=label1)
    axs[3].plot(x_ticks, stats2.errors, label=label2)
    axs[3].plot(x_ticks, stats3.errors, label=label3)
    axs[3].plot(x_ticks, stats4.errors, label=label4)
    axs[3].legend()

    os.system('spd-say "your program has finished"')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    runs = 1
    n_episodes = 10_000

    # set render_mode to "Human" to
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    n_obs = env.observation_space.n
    n_actions = env.action_space.n

    # generate_baseline(env, n_episodes=30_000, name="frozen_lake_4by4_no_slippery")
    train_4(n_obs, n_actions, runs, n_episodes, deserialize_values(name="frozen_lake_4by4_no_slippery"))
