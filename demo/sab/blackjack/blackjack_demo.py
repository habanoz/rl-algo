import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agents.a_agent import AAgent
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
from agents.wappers.state_wrapper_agent import StateWrapperAgent
from demo.sab.blackjack.blackjack_state_flattener import BlackjackStateFlattener, N_DEALER_STATES, N_PLAYER_STATES, \
    N_ACE_STATES
from model.agent_config import AgentConfig
from plot.grid_state_value_plotter import BlackjackStatePlotter


def train(env: Env, agent: AAgent, n_episodes):
    cnt = 0
    ace = 0
    for episode in tqdm(range(n_episodes)):
        c, a = play(agent, env)
        cnt += c
        ace += a

    print(ace / cnt)


def play(agent, env):
    s = env.reset()[0]  # (20,1,False)
    p, _, _ = s
    while p < 12:
        s = env.reset()[0]
        p, _, _ = s

    done = False

    cnt = 0
    ace = 0

    while not done:
        a = agent.get_action(s)

        sp, r, done, _, _ = env.step(a)

        _, _, ace_ = sp
        if ace_:
            ace += 1
        cnt += 1

        agent.update(s, a, r, done, sp)

        s = sp

    return cnt, ace


def start():
    env = gym.make("Blackjack-v1", sab=True)
    n_episodes = 551_000
    cfg = AgentConfig(epsilon=0.1, epsilon_decay=None, min_epsilon=0, gamma=1.0, alpha=0.01)
    agent = OnPolicyFirstVisitMcAgent(N_DEALER_STATES * N_PLAYER_STATES * N_ACE_STATES, 2, cfg)
    flattener = BlackjackStateFlattener()
    agentw = StateWrapperAgent(agent, flattener)

    plotter = BlackjackStatePlotter("Blackjack", 2, 2)

    train(env, agentw, 10_000)
    state_grid = agentw.state_values()
    policy_grid = agentw.get_policy()

    plotter.create_plot(state_grid[:, :, 1], policy_grid[:, :, 1], "10.000 steps(Usable Ace)", 0)
    plotter.create_plot(state_grid[:, :, 0], policy_grid[:, :, 0], "10.000 steps(No Usable Ace)", 2)

    train(env, agentw, n_episodes - 10_000)
    state_grid = agentw.state_values()
    policy_grid = agentw.get_policy()

    plotter.create_plot(state_grid[:, :, 1], policy_grid[:, :, 1], f"{n_episodes} steps(Usable Ace)", 1)
    plotter.create_plot(state_grid[:, :, 0], policy_grid[:, :, 0], f"{n_episodes} steps(No Usable Ace)", 3)

    print(flattener.coverage)
    print(flattener.ace_sums)

    plotter.show()


if __name__ == '__main__':
    start()
