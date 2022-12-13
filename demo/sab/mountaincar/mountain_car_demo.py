import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.base_agent import BaseAgent
from agents.onapp.episodic_semi_gradient_sarsa_agent import EpisodicSemiGradientSarsaAgent
from model.agent_training_config import AgentTrainingConfig


def train(n_episodes, env: gym.Env, agent: BaseAgent):
    for episode in tqdm(range(n_episodes)):
        play(agent, env)


def play(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)

        ns, r, done, _, _ = env.step(a)

        agent.update(s, a, r, done, ns)

        s = ns


def start():
    n_episodes = 1000
    env = gym.make('MountainCar-v0')
    cfg = AgentTrainingConfig(epsilon=0, gamma=1.0, alpha=0.1)
    speed_scale = 8 / (0.6 + 1.2)
    velocity_scale = 8 / (0.07 + 0.07)

    agent = EpisodicSemiGradientSarsaAgent(cfg, 3, 4096, np.array([speed_scale, velocity_scale]), 8)

    #train(n_episodes, env, agent)

    plot_episodes = [0, 99, n_episodes - 1]
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i + 1, projection='3d') for i in range(len(plot_episodes))]
    num_of_tilings = 8
    alpha = 0.3

    for ep in tqdm(range(n_episodes)):
        play(agent, env)
        if ep in plot_episodes:
            print_cost(agent, ep, axes[plot_episodes.index(ep)])

    #plt.savefig('../images/figure_10_1.png')
    plt.show()
    plt.close()


def print_cost(agent, episode, ax):
    grid_size = 40
    positions = np.linspace(-1.2, 0.6, grid_size)
    # positionStep = (POSITION_MAX - POSITION_MIN) / grid_size
    # positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    # velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / grid_size
    # velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    velocities = np.linspace(-0.07, 0.07, grid_size)
    axis_x = []
    axis_y = []
    axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(cost_to_go(position, velocity, agent))

    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))


def cost_to_go(position, velocity, agent: EpisodicSemiGradientSarsaAgent):
    costs = []
    for action in [0, 1, 2]:
        costs.append(agent.value_estimate([position, velocity], action))
    return -np.max(costs)


if __name__ == '__main__':
    start()
