import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from agents.frozen_lake_plotter import FrozenLakePlotter
from agents.mc.off_policy_mc_agent import OffPolicyMcAgent
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from agents.n_step.n_step_tree_backup_agent import NStepTreeBackupAgent
from agents.n_step.off_policy_n_step_q_sigma_agent import OffPolicyNStepQSigmaAgent
from agents.n_step.off_policy_n_step_sarsa_agent import OffPolicyNStepSarsaAgent
from agents.planning.tabular_dyna_q_agent import TabularDynaQAgent
from agents.td.double_q_learning_agent import DoubleQLearningAgent
from agents.td.expected_sarsa_agent import ExpectedSarsaAgent
from agents.td.q_learning_agent import QLearningAgent
from agents.td.sarsa_agent import SarsaAgent
from episodes_stats import EpisodesStats
from model.agent_config import AgentConfig
from model.episode_stats_wrapper import EpisodeStatsWrapper
from rms_errors_for_baseline import RMSErrorsForBaseline
from util.serialize_helper import serialize_values, deserialize_values


def generate_baseline(env, name: str, n_episodes=1000):
    agent = OnPolicyFirstVisitMcAgent(n_obs, n_actions)

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
    value_errors = np.empty(n_episodes)
    training_errors = np.empty(n_episodes)

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

        value_errors[episode] = RMSErrorsForBaseline(value_baseline).calculate_for_state(agent.state_values())

        training_errors[episode] = agent.total_training_error

    return value_errors, training_errors


def execute(agent_factory, env, n_runs=10, n_episodes=10_000, value_baseline=None):
    rolling_length = max(n_episodes // 100, 1)

    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    rms_errors = np.zeros(n_episodes)
    training_errors = np.zeros(n_episodes)
    for run in range(n_runs):
        agent = agent_factory()

        new_rms_errors, new_training_errors = generate_episodes(w_env, agent, n_episodes=n_episodes,
                                                                value_baseline=value_baseline)

        rms_errors += new_rms_errors
        training_errors += new_training_errors

        if run == runs - 1:
            FrozenLakePlotter(agent.Q, 4, 4, agent.get_desc()).show()

    rms_errors = rms_errors / runs
    training_errors = training_errors / runs

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

    training_error_moving_average = (
            np.convolve(
                training_errors, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    return EpisodesStats(reward_moving_average, cum_rewards_means, length_moving_average,
                         rms_error_moving_average, training_error_moving_average)


def train_4(n_obs: int, n_actions: int, runs: int, n_episodes: int, value_baseline: ndarray = None):
    label1 = "sarsa"
    label2 = "QL"
    label3 = "D-QL"
    label4 = "MC"

    cfg = AgentConfig(epsilon=1.0, epsilon_decay=0.99 / n_episodes, min_epsilon=0.01, alpha=0.05, gamma=0.9)

    # agent 1
    agent1 = lambda: SarsaAgent(n_obs, n_actions, cfg)
    stats1 = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 2
    agent2 = lambda: QLearningAgent(n_obs, n_actions, cfg)
    stats2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 3
    agent3 = lambda: DoubleQLearningAgent(n_obs, n_actions, cfg)
    stats3 = execute(agent3, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 4
    agent4 = lambda: OnPolicyFirstVisitMcAgent(n_obs, n_actions, cfg)
    stats4 = execute(agent4, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

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
    axs[3].plot(x_ticks, stats1.value_errors, label=label1)
    axs[3].plot(x_ticks, stats2.value_errors, label=label2)
    axs[3].plot(x_ticks, stats3.value_errors, label=label3)
    axs[3].plot(x_ticks, stats4.value_errors, label=label4)
    axs[3].legend()

    axs[4].set_title("Training Errors")
    axs[4].plot(x_ticks, stats1.training_errors, label=label1)
    axs[4].plot(x_ticks, stats2.training_errors, label=label2)
    axs[4].plot(x_ticks, stats3.training_errors, label=label3)
    axs[4].plot(x_ticks, stats4.training_errors, label=label4)
    axs[4].legend()

    plt.tight_layout()
    plt.show()


def train_4_n_td(cfg: AgentConfig, n_obs: int, n_actions: int, runs: int, n_episodes: int,
                 value_baseline: ndarray = None):
    label1 = "sarsa"
    label2 = "TB n-3"
    label3 = "off-MC"
    label4 = "Off Policy Sarsa n-3"

    # agent 1
    agent1 = lambda: SarsaAgent(n_obs, n_actions, cfg)
    stats1 = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 2
    agent2 = lambda: NStepTreeBackupAgent(n_obs, n_actions, cfg, n_step_size=3)
    stats2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 3
    agent3 = lambda: OffPolicyMcAgent(n_obs, n_actions, cfg)
    stats3 = execute(agent3, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 4
    # agent4 = lambda: OnPolicyFirstVisitMcAgent(n_obs, n_actions, cfg)
    agent4 = lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3)
    stats4 = execute(agent4, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

    x_ticks = range(len(stats1.rewards))

    axs[0].set_title("Episode rewards")
    axs[0].plot(x_ticks, stats1.rewards, label=label1)
    axs[0].plot(x_ticks, stats2.rewards, label=label2)
    axs[0].plot(x_ticks, stats3.rewards, label=label3)
    axs[0].plot(x_ticks, stats4.rewards, label=label4)
    axs[0].legend()

    axs[1].set_title("Cumulative Rewards")
    axs[1].plot(range(len(stats1.cum_rewards)), stats1.cum_rewards, label=label1)
    axs[1].plot(range(len(stats2.cum_rewards)), stats2.cum_rewards, label=label2)
    axs[1].plot(range(len(stats3.cum_rewards)), stats3.cum_rewards, label=label3)
    axs[1].plot(range(len(stats4.cum_rewards)), stats4.cum_rewards, label=label4)
    axs[1].legend()

    axs[2].set_title("Episode Lengths")
    axs[2].plot(x_ticks, stats1.lengths, label=label1)
    axs[2].plot(x_ticks, stats2.lengths, label=label2)
    axs[2].plot(x_ticks, stats3.lengths, label=label3)
    axs[2].plot(x_ticks, stats4.lengths, label=label4)
    axs[2].legend()

    axs[3].set_title("RMS Errors")
    axs[3].plot(x_ticks, stats1.value_errors, label=label1)
    axs[3].plot(x_ticks, stats2.value_errors, label=label2)
    axs[3].plot(x_ticks, stats3.value_errors, label=label3)
    axs[3].plot(x_ticks, stats4.value_errors, label=label4)
    axs[3].legend()

    axs[4].set_title("Training Errors")
    axs[4].plot(x_ticks, stats1.training_errors, label=label1)
    axs[4].plot(x_ticks, stats2.training_errors, label=label2)
    axs[4].plot(x_ticks, stats3.training_errors, label=label3)
    axs[4].plot(x_ticks, stats4.training_errors, label=label4)
    axs[4].legend()

    plt.tight_layout()
    plt.show()


def train_2_n_td(cfg: AgentConfig, n_obs: int, n_actions: int, runs: int, n_episodes: int,
                 value_baseline: ndarray = None):
    label1 = "QL"
    label2 = "Off Policy Sarsa n-3"

    # agent 1
    agent1 = lambda: QLearningAgent(n_obs, n_actions, cfg)
    stats1 = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    # agent 2
    agent2 = lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3)
    stats2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

    x_ticks = range(len(stats1.rewards))

    axs[0].set_title("Episode rewards")
    axs[0].plot(x_ticks, stats1.rewards, label=label1)
    axs[0].plot(x_ticks, stats2.rewards, label=label2)
    axs[0].legend()

    axs[1].set_title("Cumulative Rewards")
    axs[1].plot(range(len(stats1.cum_rewards)), stats1.cum_rewards, label=label1)
    axs[1].plot(range(len(stats2.cum_rewards)), stats2.cum_rewards, label=label2)
    axs[1].legend()

    axs[2].set_title("Episode Lengths")
    axs[2].plot(x_ticks, stats1.lengths, label=label1)
    axs[2].plot(x_ticks, stats2.lengths, label=label2)
    axs[2].legend()

    axs[3].set_title("RMS Errors")
    axs[3].plot(x_ticks, stats1.value_errors, label=label1)
    axs[3].plot(x_ticks, stats2.value_errors, label=label2)
    axs[3].legend()

    axs[4].set_title("Training Errors")
    axs[4].plot(x_ticks, stats1.training_errors, label=label1)
    axs[4].plot(x_ticks, stats2.training_errors, label=label2)
    axs[4].legend()

    plt.tight_layout()
    plt.show()


def train_1(cfg: AgentConfig, n_obs: int, n_actions: int, runs: int, n_episodes: int,
            value_baseline: ndarray = None):
    label1 = "sarsa"

    # agent 1
    agent1 = lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3)
    stats1 = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline)

    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

    x_ticks = range(len(stats1.rewards))

    axs[0].set_title("Episode rewards")
    axs[0].plot(x_ticks, stats1.rewards, label=label1)
    axs[0].legend()

    axs[1].set_title("Cumulative Rewards")
    axs[1].plot(range(len(stats1.cum_rewards)), stats1.cum_rewards, label=label1)
    axs[1].legend()

    axs[2].set_title("Episode Lengths")
    axs[2].plot(x_ticks, stats1.lengths, label=label1)
    axs[2].legend()

    axs[3].set_title("RMS Errors")
    axs[3].plot(x_ticks, stats1.value_errors, label=label1)
    axs[3].legend()

    axs[4].set_title("Training Errors")
    axs[4].plot(x_ticks, stats1.training_errors, label=label1)
    axs[4].legend()

    plt.tight_layout()
    plt.show()


def train_agents(agents, labels, runs: int, n_episodes: int, value_baseline: ndarray = None):
    stats = []

    for agent in agents:
        stats.append(execute(agent, env, n_runs=runs, n_episodes=n_episodes, value_baseline=value_baseline))

    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

    axs[0].set_title("Episode rewards")
    for stat, label in zip(stats, labels):
        axs[0].plot(range(len(stat.rewards)), stat.rewards, label=label)
    axs[0].legend()

    axs[1].set_title("Cumulative Rewards")
    for stat, label in zip(stats, labels):
        axs[1].plot(range(len(stat.cum_rewards)), stat.cum_rewards, label=label)
    axs[1].legend()

    axs[2].set_title("Episode Lengths")
    for stat, label in zip(stats, labels):
        axs[2].plot(range(len(stat.lengths)), stat.lengths, label=label)
    axs[2].legend()

    axs[3].set_title("RMS Errors")
    for stat, label in zip(stats, labels):
        axs[3].plot(range(len(stat.value_errors)), stat.value_errors, label=label)
    axs[3].legend()

    axs[4].set_title("Training Errors")
    for stat, label in zip(stats, labels):
        axs[4].plot(range(len(stat.training_errors)), stat.training_errors, label=label)
    axs[4].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    runs = 1
    n_episodes = 3_000

    # set render_mode to "Human" to
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    n_obs = env.observation_space.n
    n_actions = env.action_space.n

    # cfg = AgentConfig(epsilon=1.0, epsilon_decay=0.99 / n_episodes, min_epsilon=0.01, alpha=0.05, gamma=0.9)
    cfg = AgentConfig(epsilon=0.4, epsilon_decay=None, min_epsilon=0.01, alpha=0.9, gamma=0.9)

    # generate_baseline(env, n_episodes=30_000, name="frozen_lake_4by4_no_slippery")

    agents = [
        # lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=1),
        # lambda: NStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=1),
        # lambda: ExpectedSarsaAgent(n_obs, n_actions, cfg),
        # lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3),
        # lambda: NStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3),
        # lambda: SarsaAgent(n_obs, n_actions, cfg),
        # lambda: QLearningAgent(n_obs, n_actions, cfg),
        # lambda: NStepTreeBackupAgent(n_obs, n_actions, cfg, n_step_size=1),
        # lambda: NStepTreeBackupAgent(n_obs, n_actions, cfg, n_step_size=3),
        lambda: OffPolicyNStepQSigmaAgent(n_obs, n_actions, cfg, n_step_size=1)
    ]

    labels = [
        # "OffPolicyNStepSarsaAgent n-1",
        # "NStepSarsaAgent n-1",
        # "ExpectedSarsa",
        # "OffPolicyNStepSarsaAgent n-3",
        # "NStepSarsaAgent n-3",
        # "QL",
        # "TB-1",
        # "TB-3",
        "Q Sigma-n1"
    ]
    train_agents(agents, labels, runs, n_episodes, deserialize_values(name="frozen_lake_4by4_no_slippery"))
