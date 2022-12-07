import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from env.random_walk_env import RandomWalkEnv


def generate_episodes(env, agent, n_episodes=1000):
    episode_errors = np.empty(n_episodes)
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

        values = np.array([np.mean(r) for r in agent.Q])
        episode_errors[episode] = np.sqrt(np.sum(np.power(values - env.true_values, 2)) / len(env.true_values))

    return episode_errors


def execute(agent_factory, env, n_runs=10, n_episodes=10_000, rolling_length=100):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)
    errors = np.empty((n_runs, n_episodes))

    for run in range(n_runs):
        agent = agent_factory()

        episode_errors = generate_episodes(w_env, agent, n_episodes=n_episodes)

        errors[run, :] = episode_errors

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

    error_means = np.mean(errors, axis=0)
    errors_moving_average = (
            np.convolve(
                error_means, np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )

    return reward_moving_average, length_moving_average, errors_moving_average


def start():
    runs = 5
    n_episodes = 200

    env = RandomWalkEnv()
    n_obs = env.observation_space.n
    n_actions = env.action_space.n

    agent1 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=None, alpha=0.3,
                                     n_step_size=1)
    reward_moving_average, length_moving_average, errors_moving_average = execute(agent1, env, n_runs=runs, n_episodes=n_episodes, rolling_length=1)

    agent2 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=None, alpha=0.3,
                                     n_step_size=2)
    reward_moving_average2, length_moving_average2, errors_moving_average2 = execute(agent2, env, n_runs=runs, n_episodes=n_episodes, rolling_length=1)

    agent3 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=None, alpha=0.3,
                                     n_step_size=3)
    reward_moving_average3, length_moving_average3, errors_moving_average3 = execute(agent3, env, n_runs=runs, n_episodes=n_episodes, rolling_length=1)

    agent4 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=None, alpha=0.3,
                                     n_step_size=4)
    reward_moving_average4, length_moving_average4, errors_moving_average4 = execute(agent4, env, n_runs=runs, n_episodes=n_episodes, rolling_length=1)

    agent5 = lambda: NStepSarsaAgent(n_obs, n_actions, epsilon=1.0, epsilon_decay=None, alpha=0.3,
                                     n_step_size=5)
    reward_moving_average5, length_moving_average5, errors_moving_average5 = execute(agent5, env, n_runs=runs, n_episodes=n_episodes, rolling_length=1)

    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(20, 10))
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

    axs[2].set_title("Episode Errors")
    axs[2].plot(range(len(errors_moving_average)), errors_moving_average, label="n-1")
    axs[2].plot(range(len(errors_moving_average2)), errors_moving_average2, label="n-2")
    axs[2].plot(range(len(errors_moving_average3)), errors_moving_average3, label="n-3")
    axs[2].plot(range(len(errors_moving_average4)), errors_moving_average4, label="n-4")
    axs[2].plot(range(len(errors_moving_average5)), errors_moving_average5, label="n-5")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start()
