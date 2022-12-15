import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray

from agents.base_agent import AgentTrainingConfig, StateFeature, ValueFeature
from agents.pg.continuing_et_actor_critic_agent import ContinuingETActorCriticAgent
from env.access_control_env import AccessControlEnv
from util import tiles
from util.tiles import IHT, tiles


def train_agent(agent, env):
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.get_action(s)
        ns, r, done, _, _ = env.step(a)
        agent.update(s, a, r, done, ns)

        s = ns


class AccessControlFeature(ValueFeature):
    def __init__(self):
        self.size = 256
        self.iht = IHT(self.size)

        self.n_tilings = 4
        n_server_scale = self.n_tilings / 10
        priority_scale = self.n_tilings / 3
        self.state_scale = np.array([n_server_scale, priority_scale])

    def s_a(self, state, action) -> ndarray:
        return tiles(self.iht, self.n_tilings, state * self.state_scale, [action])

    def dim(self):
        return self.size


class AccessControlWFeature(StateFeature):
    def __init__(self):
        self.size = 128
        self.iht = IHT(self.size)

        self.n_tilings = 4
        n_server_scale = self.n_tilings / 10
        priority_scale = self.n_tilings / 3
        self.state_scale = np.array([n_server_scale, priority_scale])

    def s(self, state) -> ndarray:
        return tiles(self.iht, self.n_tilings, state * self.state_scale)

    def dim(self):
        return self.size


def start():
    env = AccessControlEnv(time_limit=500_000)

    n_servers = 10
    n_priorities = 4
    n_states = (n_servers + 1) * n_priorities

    cfg = AgentTrainingConfig(alpha=pow(2, -9), gamma=1.0, alpha_w=pow(2, -6), alpha_r=pow(2, -6), lambdaa=0.5,
                              lambda_w=0.5)
    agent = ContinuingETActorCriticAgent(n_states, 2, cfg, AccessControlFeature(), AccessControlWFeature())

    train_agent(agent, env)

    plot_for_agent(agent)

    print("done")


def plot_for_agent(agent):
    lines = np.zeros((4, 11))
    for server in range(11):
        for p in range(4):
            lines[p, server] = agent.value_estimate(np.array([server, p]))

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
        lambda obs: np.argmax(agent.pi_s([obs[1], obs[0]])),
        axis=2,
        arr=np.dstack([xx, yy]),
    )

    sns.heatmap(policy_grid, ax=axs[1], linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    axs[1].set_title(f"Policy")
    axs[1].set_xlabel("Priority")
    axs[1].set_ylabel("Free Servers")
    axs[1].set_xticklabels([1, 2, 4, 8])
    # axs[1].set_yticklabels(range(0, 4))

    plt.tight_layout()
    plt.show()

    print("done")


if __name__ == '__main__':
    start()
    # plot()
