import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.onapp.episodic_semi_gradient_n_step_sarsa_agent import EpisodicSemiGradientNStepSarsaAgent
from env.episode_stats_wrapper import EpisodeStatsWrapper
from agents.base_agent import BaseAgent, AgentTrainingConfig


def play(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)

        ns, r, done, truncated, _ = env.step(a)

        agent.update(s, a, r, done, ns, truncated)

        done = done or truncated

        s = ns


def generate_episodes(env, agent, n_episodes=1000):
    for episode in range(n_episodes):
        play(agent, env)


def execute(alpha, step, env, n_runs, n_episodes):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    n_actions = 3
    n_states = 2048
    n_tilings = 8
    speed_scale = n_tilings / (0.6 + 1.2)
    velocity_scale = n_tilings / (0.07 + 0.07)
    state_scale = np.array([speed_scale, velocity_scale])

    agent = EpisodicSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha),
                                                n_actions, n_states, state_scale, n_tilings, n_step_size=step)
    for run in tqdm(range(n_runs)):
        generate_episodes(w_env, agent, n_episodes=n_episodes)

    return np.mean(w_env.lengths)


def train_agents(alphas, steps, env: gym.Env, runs: int, n_episodes: int):
    lengths = np.empty((len(steps), len(alphas)))

    for s in range(len(steps)):
        for a in range(len(alphas)):
            length = execute(alphas[a], steps[s], env, runs, n_episodes)
            lengths[s, a] = length

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    # plt.yscale("log")
    axs.set_title("Episode Lengths")
    for l in range(len(lengths)):
        axs.plot(lengths[l], label=steps[l])
    axs.legend()
    axs.set_xticklabels(alphas)
    plt.tight_layout()
    plt.show()


def start():
    n_episodes = 50
    n_runs = 50

    env = gym.make('MountainCar-v0', max_episode_steps=300)
    step_list = [1, 2, 4, 8, 16]
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

    train_agents(alpha_list, step_list, env, n_runs, n_episodes)


if __name__ == '__main__':
    start()
