import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from agents.base_agent import AgentTrainingConfig, Feature, StateFeature, ValueFeature
from agents.pg.episodic_et_actor_critic_agent import EpisodicETActorCriticAgent
from agents.pg.et_actor_critic_agent import ETActorCriticAgent
from agents.pg.one_step_actor_critic_agent import OneStepActorCriticAgent
from agents.pg.reinforce_softmax_linear_mc_agent import ReinforceSoftmaxLinearMcAgent
from agents.pg.reinforce_softmax_linear_wih_baselined_mc_agent import ReinforceSoftmaxLinearWithBaselineMcAgent
from env.episode_stats_wrapper import EpisodeStatsWrapper
from env.short_corridor_env import ShortCorridorEnv


class CorridorFeature(Feature):
    def __init__(self):
        self.x = np.array([[0, 1], [1, 0]])

    def s_a(self, state, action) -> ndarray:
        return self.x[action]

    def s(self, state) -> ndarray:
        return self.x


class CorridorStateFeature(StateFeature):
    def __init__(self):
        self.x = np.array([[0, 1], [1, 0]])

    def s(self, state) -> ndarray:
        return self.x

    def dim(self):
        return 2


class CorridorValueFeature(ValueFeature):
    def __init__(self):
        self.x = np.array([[0, 1], [1, 0]])

    def s_a(self, state, action) -> ndarray:
        return self.x[action]

    def dim(self):
        return 2

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


def execute(agent_producer, env, n_runs, n_episodes):
    w_env = EpisodeStatsWrapper(env, n_runs=n_runs, n_episodes=n_episodes)

    for run in tqdm(range(n_runs)):
        agent = agent_producer()
        generate_episodes(w_env, agent, n_episodes=n_episodes)

    return np.mean(w_env.rewards, axis=0)


def start():
    n_runs = 50
    n_episodes = 1000

    env = ShortCorridorEnv()
    agents = [
        lambda: OneStepActorCriticAgent(4, 2, AgentTrainingConfig(alpha=pow(2, -9), gamma=1.0, alpha_w=pow(2, -6)), CorridorFeature(),initial_theta=np.array([-1.0, 1.0])),

        lambda: ETActorCriticAgent(4, 2, AgentTrainingConfig(alpha=pow(2, -9), gamma=1.0, alpha_w=pow(2, -6), lambdaa=0.5, lambda_w=0.5), CorridorFeature(),initial_theta=np.array([-1.0, 1.0])),

        lambda : EpisodicETActorCriticAgent(44, 2, AgentTrainingConfig(alpha=pow(2, -9), gamma=1.0, alpha_w=pow(2, -6), lambdaa=0.5, lambda_w=0.5), CorridorValueFeature(), CorridorStateFeature()),

        lambda: ReinforceSoftmaxLinearMcAgent( 4, 2, AgentTrainingConfig(alpha=pow(2, -12), gamma=1.0), CorridorFeature(), initial_theta=np.array([-1.0, 1.0])),

        lambda: ReinforceSoftmaxLinearWithBaselineMcAgent( 4, 2, AgentTrainingConfig(alpha=pow(2, -9), gamma=1.0, alpha_w=pow(2, -6)), CorridorFeature(), initial_theta=np.array([-1.0, 1.0])),
    ]

    agent_labels = ["Actor Critic a=2^-9,a_w=a=2^-6",
                    "ET Actor Critic a=2^-9,a_w=a=2^-6 lambda 0.9",
                    "Episodic ET Actor Critic a=2^-9,a_w=a=2^-6 lambda 0.9",
                    "REINFORCE a=2^-12",
                    "REINFORCE  with baseline a=2^-9,a_w=a=2^-6"
                    ]

    assert len(agents) == len(agent_labels)

    agent_rewards = []
    for i in range(len(agents)):
        rewards = execute(agents[i], env, n_runs, n_episodes)
        agent_rewards.append(rewards)

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))

    axs.set_title("Total Rewards (Initial theta set for greedy left actions)")

    for i in range(len(agents)):
        axs.plot(agent_rewards[i], label=agent_labels[i])

    axs.set_ylabel("Total Reward on Episode")
    axs.set_ylabel("Episode")

    axs.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start()
