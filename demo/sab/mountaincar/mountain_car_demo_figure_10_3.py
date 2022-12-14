import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from agents.onapp.episodic_semi_gradient_n_step_sarsa_agent import EpisodicSemiGradientNStepSarsaAgent
from agents.onapp.episodic_semi_gradient_sarsa_agent import EpisodicSemiGradientSarsaAgent
from env.episode_stats_wrapper import EpisodeStatsWrapper
from agents.base_agent import BaseAgent, AgentTrainingConfig


def play(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)

        ns, r, done, truncated, _ = env.step(a)

        if truncated:
            print(truncated)

        done = done or truncated

        agent.update(s, a, r, done, ns)

        s = ns


def generate_episodes(env, agent, n_episodes=1000):
    for episode in range(n_episodes):
        play(agent, env)


def execute(agent_factory, env, n_runs=10, n_episodes=10_000):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    for run in tqdm(range(n_runs)):
        agent = agent_factory()

        generate_episodes(w_env, agent, n_episodes=n_episodes)

    length_means = np.mean(w_env.lengths, axis=0)

    return length_means


def train_agents(agents, env: gym.Env, labels, runs: int, n_episodes: int, value_baseline: ndarray = None):
    stats = []

    for agent in agents:
        stats.append(execute(agent, env, n_runs=runs, n_episodes=n_episodes))

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    # plt.yscale("log")
    axs.set_title("Episode Lengths")
    for lengths, label in zip(stats, labels):
        axs.plot(range(len(lengths)), lengths, label=label)
    axs.legend()

    plt.tight_layout()
    plt.show()


def start():
    n_episodes = 500
    n_actions = 3
    n_states = 2048
    n_runs = 100

    env = gym.make('MountainCar-v0', max_episode_steps=10_000)
    n_tilings = 8
    speed_scale = n_tilings / (0.6 + 1.2)
    velocity_scale = n_tilings / (0.07 + 0.07)
    state_scale = np.array([speed_scale, velocity_scale])

    agents = [
        lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=0.1), n_actions,
                                                    n_states, state_scale, n_tilings, n_step_size=1),
        lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=0.2), n_actions,
                                                    n_states, state_scale, n_tilings, n_step_size=8)
    ]

    labels = [
        f"n=1 a=0.5/{n_tilings}",
        f"n=8 a=0.3/{n_tilings}",
    ]

    train_agents(agents, env, labels, n_runs, n_episodes)


if __name__ == '__main__':
    start()
