import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from plot.frozen_lake_plotter import FrozenLakePlotter
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
from agents.n_step.n_step_tree_backup_agent import NStepTreeBackupAgent
from agents.n_step.off_policy_n_step_q_sigma_agent import OffPolicyNStepQSigmaAgent
from episodes_stats import EpisodesStats
from model.agent_training_config import AgentTrainingConfig
from env.episode_stats_wrapper import EpisodeStatsWrapper
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
    runs = 50
    n_episodes = 1_000

    # set render_mode to "Human" to
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    n_obs = env.observation_space.n
    n_actions = env.action_space.n

    # cfg = AgentConfig(epsilon=1.0, epsilon_decay=0.99 / n_episodes, min_epsilon=0.01, alpha=0.05, gamma=0.9)
    cfg = AgentTrainingConfig(epsilon=0.5, epsilon_decay=None, min_epsilon=0.01, alpha=0.1, gamma=0.9)

    # generate_baseline(env, n_episodes=30_000, name="frozen_lake_4by4_no_slippery")

    agents = [
        # lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=1),
        # lambda: NStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=1),
        # lambda: ExpectedSarsaAgent(n_obs, n_actions, cfg),
        # lambda: OffPolicyNStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3),
        # lambda: NStepSarsaAgent(n_obs, n_actions, cfg, n_step_size=3),
        # lambda: SarsaAgent(n_obs, n_actions, cfg),
        #lambda: QLearningAgent(n_obs, n_actions, cfg),
        lambda: NStepTreeBackupAgent(n_obs, n_actions, cfg, n_step_size=1),
        lambda: NStepTreeBackupAgent(n_obs, n_actions, cfg, n_step_size=5),
        lambda: OffPolicyNStepQSigmaAgent(n_obs, n_actions, cfg, n_step_size=1),
        #lambda: OffPolicyNStepQSigmaAgent(n_obs, n_actions, cfg, n_step_size=2),
        lambda: OffPolicyNStepQSigmaAgent(n_obs, n_actions, cfg, n_step_size=5),
    ]

    labels = [
        # "OffPolicyNStepSarsaAgent n-1",
        # "NStepSarsaAgent n-1",
        # "ExpectedSarsa",
        # "OffPolicyNStepSarsaAgent n-3",
        # "NStepSarsaAgent n-3",
        #"QL",
        "TB-1",
        "TB-5",
        "Q Sigma-n1",
        #"Q Sigma-n2",
        "Q Sigma-n5",
    ]
    train_agents(agents, labels, runs, n_episodes, deserialize_values(name="frozen_lake_4by4_no_slippery"))
