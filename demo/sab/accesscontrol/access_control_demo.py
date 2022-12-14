import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from agents.onapp.differential_semi_gradient_sarsa_agent import DifferentialSemiGradientSarsaAgent
from env.access_control_env import AccessControlEnv
from agents.base_agent import BaseAgent, AgentTrainingConfig


def train_agent(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)
        ns, r, done, _, _ = env.step(a)
        agent.update(s, a, r, done, ns)

        s = ns


def start():
    env = AccessControlEnv(time_limit=2_000_000)
    n_tilings = 8
    n_server_scale = n_tilings / 10
    priority_scale = n_tilings / 3
    state_scale = np.array([n_server_scale, priority_scale])

    agent = DifferentialSemiGradientSarsaAgent(AgentTrainingConfig(epsilon=0.1, alpha=0.01, beta=0.01, gamma=1.0), 2,
                                               1024, state_scale, n_tilings)

    #agent = DifferentialSemiGradientNStepSarsaAgent(AgentTrainingConfig(epsilon=0.1, alpha=0.01, beta=0.01, gamma=1.0),
    #                                                2, 1024, state_scale, n_tilings)

    train_agent(agent, env)

    plot_for_agent(agent)
    print("done")


def plot_for_agent(agent):
    lines = np.zeros((4, 11))
    for server in range(11):
        for p in range(4):
            lines[p, server] = max(agent.value_estimate(np.array([server, p]), 0),
                                   agent.value_estimate(np.array([server, p]), 1))

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))

    axs[0].set_title("Episode rewards")
    for stat, label in zip(lines, ["Priority 1", "Priority 2", "Priority 4", "Priority 8"]):
        axs[0].plot(range(len(stat)), stat, label=label)
    axs[0].legend()

    xx, yy = np.meshgrid(
        np.arange(0, 4),
        np.arange(0, 11),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: np.argmax([agent.value_estimate(np.array([obs[1], obs[0]]), 0),
                               agent.value_estimate(np.array([obs[1], obs[0]]), 1)]),
        axis=2,
        arr=np.dstack([xx, yy]),
    )

    sns.heatmap(policy_grid, ax=axs[1], linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    axs[1].set_title(f"Policy")
    axs[1].set_xlabel("Priority")
    axs[1].set_ylabel("Free Servers")
    axs[1].set_xticklabels([1, 2, 4, 8])
    # axs[1].set_yticklabels(range(0, 4))

    print([(agent.value_estimate(np.array([a, 0]), 0), agent.value_estimate(np.array([a, 0]), 1)) for a in range(11)])
    plt.tight_layout()
    plt.show()

    print("done")


if __name__ == '__main__':
    start()
    # plot()
