import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from agents.base_agent import AgentTrainingConfig, Feature
from agents.et.pg.reinforce_softmax_linear_mc_agent import ReinforceSoftmaxLinearMcAgent
from env.episode_stats_wrapper import EpisodeStatsWrapper
from env.short_corridor_env import ShortCorridorEnv


class CorridorFeature(Feature):
    def __init__(self):
        self.x = np.array([[0, 1], [1, 0]])

    def s_a(self, state, action) -> ndarray:
        return self.x[action]

    def s(self, state) -> ndarray:
        return self.x


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


def execute(alpha, env, n_runs, n_episodes):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    for run in tqdm(range(n_runs)):
        agent = ReinforceSoftmaxLinearMcAgent(4, 2, AgentTrainingConfig(alpha=alpha, gamma=1.0),
                                              CorridorFeature(), initial_theta=np.array([-1.0, 1.0]))
        generate_episodes(w_env, agent, n_episodes=n_episodes)

    return np.mean(w_env.rewards, axis=0)


def start():
    n_runs = 100
    n_episodes = 1000

    env = ShortCorridorEnv()
    alphas = [pow(2, -12), pow(2, -13), pow(2, -14)]
    alpha_labels = ["2^-12", "2^-13", "2^-14"]

    assert len(alphas) == len(alpha_labels)

    agent_rewards = []
    for i in range(len(alphas)):
        rewards = execute(alphas[i], env, n_runs, n_episodes)
        agent_rewards.append(rewards)

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))

    axs.set_title("Total Rewards (Initial theta set for greedy left actions)")

    for i in range(len(alphas)):
        axs.plot(agent_rewards[i], label=alpha_labels[i])

    axs.set_ylabel("Total Reward on Episode")
    axs.set_ylabel("Episode")

    axs.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start()
