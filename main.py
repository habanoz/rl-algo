import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from pandas._libs.parsers import defaultdict
from tqdm import tqdm

from EpisodeStatsWrapper import EpisodeStatsWrapper
from agents.td.q_learning_agent import QLearningAgent
from agents.td.sarsa_agent import SarsaAgent


def train_agent(env, agent, n_episodes=1000):
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

def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    # ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # ax1.plot_surface(
    #     cols,
    #     rows,
    #     value,
    #     rstride=1,
    #     cstride=1,
    #     cmap="viridis",
    #     edgecolor="none",
    # )
    # plt.xticks(range(0, 4), range(0, 4))
    # plt.yticks(range(0, 4), range(0, 4))
    # ax1.set_title(f"State values: {title}")
    # ax1.set_xlabel("Cols")
    # ax1.set_ylabel("Rows")
    # ax1.zaxis.set_rotate_label(False)
    # ax1.set_zlabel("Value", fontsize=14, rotation=90)
    # ax1.view_init(45, 150)

    # plot the policy
    fig.add_subplot(1, 2, 1)
    ax1 = sns.heatmap(value_grid, linewidth=0, annot=True, cmap="cool", cbar=False)
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Cols")
    ax1.set_ylabel("Rows")
    ax1.set_xticklabels(range(0, 4))
    ax1.set_yticklabels(range(0, 4))

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Cols")
    ax2.set_ylabel("Rows")
    ax2.set_xticklabels(range(0, 4))
    ax2.set_yticklabels(range(0, 4))

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Left"),
        Patch(facecolor="grey", edgecolor="black", label="Down"),
        Patch(facecolor="red", edgecolor="black", label="Right"),
        Patch(facecolor="blue", edgecolor="black", label="Up"),
    ]
    ax2.legend()
    return fig

def create_grids(Q, n_rows, n_cols):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in enumerate(Q):
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    cols, rows = np.meshgrid(
        np.arange(0, n_cols),
        np.arange(0, n_rows),
    )

    # create the value grid for plotting
    value_grid = np.apply_along_axis(
        lambda obs: state_value[obs[0]*4+obs[1]],
        axis=2,
        arr=np.dstack([rows, cols]),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[obs[0]*4+obs[1]],
        axis=2,
        arr=np.dstack([rows, cols]),
    )
    return value_grid, policy_grid

if __name__ == '__main__':
    runs = 5
    n_episodes = 10_000
    demo_runs = False

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
                   render_mode=None)  # set render_mode to "Human"
    env = EpisodeStatsWrapper(env, n_runs=runs, n_episodes=n_episodes)

    for run in range(runs):
        agent = QLearningAgent(env.observation_space.n, env.action_space.n, epsilon=1.0, epsilon_decay=0.99 / n_episodes)

        train_agent(env, agent, n_episodes=n_episodes)

        if demo_runs and run == runs - 1:
            demo_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
            agent.epsilon = 0.0
            agent.epsilon_decay = None
            # train_agent(demo_env, agent, n_episodes=10)

            value_grid, policy_grid = create_grids(agent.Q, 4, 4)
            fig1 = create_plots(value_grid, policy_grid, title="")
            plt.show()

            print("asdasd")

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
    axs[0].set_title("Episode rewards")
    rewards_means = np.mean(env.rewards, axis=0)

    axs[0].plot(range(n_episodes), rewards_means)

    axs[1].set_title("Episode lengths")
    length_means = np.mean(env.lengths, axis=0)
    axs[1].plot(range(n_episodes), length_means)


    plt.tight_layout()
    plt.show()

