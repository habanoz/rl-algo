import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agents.a_agent import AAgent
from agents.balckjack_agent import BlackjackAgent
from plot.grid_state_value_plotter_copied import BlackjackStatePlotterCopied


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

    learning_rate = 0.01
    n_episodes = 502_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    agent = BlackjackAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    train(env, agent, 10_000)
    state_grid = agent.state_values()
    policy_grid = agent.get_policy()

    plotter = BlackjackStatePlotterCopied("asdasdasdasdsa", 2, 4)

    plotter.create_plot(state_grid, policy_grid, "10.000 steps(Usable Ace)", 0, usable_ace=True)
    plotter.create_plot(state_grid, policy_grid, "10.000 steps(No Usable Ace)", 2, usable_ace=False)

    train(env, agent, n_episodes - 10_000)
    state_grid = agent.state_values()
    policy_grid = agent.get_policy()

    plotter.create_plot(state_grid, policy_grid, f"{n_episodes} steps(Usable Ace)", 1, usable_ace=True)
    plotter.create_plot(state_grid, policy_grid, f"{n_episodes} steps(No Usable Ace)", 3, usable_ace=False)

    plotter.show()


if __name__ == '__main__':
    start()
