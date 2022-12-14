import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.et.bf_linear_fa_sarsa_agent import BFLinearFASarsaAgent
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


def execute(alpha, lambdaa, env, n_runs, n_episodes):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    n_actions = 3
    n_states = 2048
    n_tilings = 8
    speed_scale = n_tilings / (0.6 + 1.2)
    velocity_scale = n_tilings / (0.07 + 0.07)
    state_scale = np.array([speed_scale, velocity_scale])

    agent = BFLinearFASarsaAgent(AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=alpha, lambdaa=lambdaa), n_actions, n_states, state_scale, n_tilings)
    for run in tqdm(range(n_runs)):
        generate_episodes(w_env, agent, n_episodes=n_episodes)

    return np.mean(w_env.lengths)


def train_agents(alphas, lambdas, env: gym.Env, runs: int, n_episodes: int):
    lengths = np.empty((len(lambdas), len(alphas)))

    max_length = 0
    for s in range(len(lambdas)):
        for a in range(len(alphas)):
            if lambdas[s] > 0.96 and alphas[a] >= 0.8:
                # within this range: episode lengths exceed max limit and truncates
                # long episodes also distorts the chart
                length = execute(alphas[a], lambdas[s], env, runs, n_episodes)
                length = min(length, max_length)
            else:
                length = execute(alphas[a], lambdas[s], env, runs, n_episodes)
                max_length = max(max_length, length)

            lengths[s, a] = length

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    # plt.yscale("log")
    axs.set_title("Episode Lengths")
    for l in range(len(lengths)):
        axs.plot(lengths[l], label=lambdas[l])

    axs.set_xlabel("Alpha")
    axs.set_ylabel("Average Episode Length")

    axs.set_xticklabels([0]+alphas)
    axs.legend()

    plt.tight_layout()
    plt.show()


def start():
    n_episodes = 50
    n_runs = 5

    env = gym.make('MountainCar-v0', max_episode_steps=300)
    lambda_list = [0.0, 0.68, 0.84, 0.92, 0.99]
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

    train_agents(alpha_list, lambda_list, env, n_runs, n_episodes)


if __name__ == '__main__':
    start()
