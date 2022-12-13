import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.onapp.episodic_semi_gradient_sarsa_agent import EpisodicSemiGradientSarsaAgent
from model.agent_training_config import AgentTrainingConfig


def cost_to_go(position, velocity, agent: EpisodicSemiGradientSarsaAgent):
    costs = []
    for action in [0, 1, 2]:
        costs.append(agent.value_estimate([position, velocity], action))
    return -np.max(costs)


def print_cost(agent, episode, ax):
    grid_size = 40
    positions = np.linspace(-1.2, 0.6, grid_size)
    velocities = np.linspace(-0.07, 0.07, grid_size)

    pp, vv = np.meshgrid(
        positions,
        velocities,
    )

    value = np.apply_along_axis(
        lambda obs: cost_to_go(obs[0], obs[1], agent),
        axis=2,
        arr=np.dstack([pp, vv]),
    )

    ax.plot_surface(pp, vv, value, cmap="viridis")
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))


def play(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)

        ns, r, done, _, _ = env.step(a)

        agent.update(s, a, r, done, ns)

        s = ns


def start():
    n_episodes = 10_000
    env = gym.make('MountainCar-v0')
    cfg = AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=0.1)
    n_tilings = 8
    speed_scale = n_tilings / (0.6 + 1.2)
    velocity_scale = n_tilings / (0.07 + 0.07)

    agent = EpisodicSemiGradientSarsaAgent(cfg, 3, 2048, np.array([speed_scale, velocity_scale]), n_tilings)

    plot_episodes = [0, 99, 999, n_episodes - 1]
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i + 1, projection='3d') for i in range(len(plot_episodes))]

    for ep in tqdm(range(n_episodes)):
        play(agent, env)
        if ep in plot_episodes:
            print_cost(agent, ep, axes[plot_episodes.index(ep)])

    # plt.savefig('../../../images/figure_10.1.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    start()
