import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agents.base_agent import AgentTrainingConfig, AAgent
from agents.planning.tabular_dyna_q_agent import TabularDynaQAgent
from agents.wrappers.state_wrapper_agent import StateWrapperAgent
from demo.sab.blackjack.blackjack_state_flattener import BlackjackStateFlattener, N_DEALER_STATES, N_PLAYER_STATES, \
    N_ACE_STATES
from plot.grid_state_value_plotter import BlackjackStatePlotter


def train(env: Env, agent: AAgent, n_episodes):
    for episode in tqdm(range(n_episodes)):
        play(agent, env)


def play(agent, env):
    s = env.reset()[0]  # (20,1,False)
    p, _, _ = s

    while p < 12:  # player sum range should be [12,21]
        s = env.reset()[0]
        p, _, _ = s

    done = False

    while not done:
        a = agent.get_action(s)

        sp, r, done, _, _ = env.step(a)

        agent.update(s, a, r, done, sp)

        s = sp


def start():
    env = gym.make("Blackjack-v1", sab=True)
    n_episodes = 150_000
    cfg = AgentTrainingConfig(epsilon=0.5, epsilon_decay=0.5/(n_episodes/2), min_epsilon=0.01, gamma=1.0, alpha=0.01)
    agent = TabularDynaQAgent(N_DEALER_STATES * N_PLAYER_STATES * N_ACE_STATES, 2, cfg)
    flattener = BlackjackStateFlattener()
    agentw = StateWrapperAgent(agent, flattener)

    plotter = BlackjackStatePlotter("Blackjack-MC", 2, 1)

    train(env, agentw, n_episodes)
    state_grid = agentw.state_values_max()
    policy_grid = agentw.get_policy()

    plotter.create_plot(state_grid[:, :, 1], policy_grid[:, :, 1], f"{n_episodes} steps(Usable Ace)", 0)
    plotter.create_plot(state_grid[:, :, 0], policy_grid[:, :, 0], f"{n_episodes} steps(No Usable Ace)", 1)

    print(flattener.coverage)
    print(flattener.ace_sums)

    plotter.show()
    #plotter.save("blackjack-MC-6_000_000")


if __name__ == '__main__':
    start()
