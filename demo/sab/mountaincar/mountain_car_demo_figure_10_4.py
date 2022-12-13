import gymnasium as gym
import numpy
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from agents.onapp.episodic_semi_gradient_n_step_sarsa_agent import EpisodicSemiGradientNStepSarsaAgent
from agents.onapp.episodic_semi_gradient_sarsa_agent import EpisodicSemiGradientSarsaAgent
from env.episode_stats_wrapper import EpisodeStatsWrapper
from model.agent_training_config import AgentTrainingConfig


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


def execute_agents(agents, env, n_runs=10, n_episodes=10_000):
    stats = []
    for agent in agents:
        lengths = execute(agent, env, n_runs, n_episodes)
        stats.append(np.mean(lengths))

    return np.array(stats)


def train_agents(agents_list, env: gym.Env, labels, runs: int, n_episodes: int):
    stats = []

    for agents, label in zip(agents_list, labels):
        print(f"starting {label}")
        stat = execute_agents(agents, env, n_runs=runs, n_episodes=n_episodes)
        stats.append(stat)
        print(f"completed {label}")

        numpy.savetxt(f"array_{label}.txt", stat)

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    # plt.yscale("log")
    axs.set_title("Episode Lengths")
    for lengths, label in zip(stats, labels):
        axs.plot(range(len(lengths)), lengths, label=label)
    axs.legend()

    plt.tight_layout()
    plt.show()


def start():
    n_episodes = 50
    n_actions = 3
    n_states = 2048
    n_runs = 100

    env = gym.make('MountainCar-v0', max_episode_steps=10_000)
    n_tilings = 8
    speed_scale = n_tilings / (0.6 + 1.2)
    velocity_scale = n_tilings / (0.07 + 0.07)
    state_scale = np.array([speed_scale, velocity_scale])

    alpha_list = np.linspace(0.1, 1.6, 16)

    agents = [
        # [lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha), n_actions,
        #                                              n_states, state_scale, n_tilings, n_step_size=1)
        #  for alpha in alpha_list],
        # [lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha), n_actions,
        #                                              n_states, state_scale, n_tilings, n_step_size=2)
        #  for alpha in alpha_list],
        # [lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha), n_actions,
        #                                              n_states, state_scale, n_tilings, n_step_size=4)
        #  for alpha in alpha_list],
        [lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha), n_actions,
                                                     n_states, state_scale, n_tilings, n_step_size=8)
         for alpha in alpha_list],
        [lambda: EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha), n_actions,
                                                     n_states, state_scale, n_tilings, n_step_size=16)
         for alpha in alpha_list]
    ]

    labels = [
        #f"n=1",
        #f"n=2",
        #f"n=4",
        f"n=8",
        f"n=16",
    ]

    train_agents(agents, env, labels, n_runs, n_episodes)


if __name__ == '__main__':
    start()
