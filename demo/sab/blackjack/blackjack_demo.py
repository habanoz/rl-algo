import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agents.a_agent import AAgent
from agents.mc.off_policy_mc_agent import OffPolicyMcAgent
from agents.mc.on_policy_first_visit_mc_agent import OnPolicyFirstVisitMcAgent
from agents.n_step.n_step_sarsa_agent import NStepSarsaAgent
from agents.n_step.n_step_tree_backup_agent import NStepTreeBackupAgent
from agents.n_step.off_policy_n_step_q_sigma_agent import OffPolicyNStepQSigmaAgent
from agents.n_step.off_policy_n_step_sarsa_agent import OffPolicyNStepSarsaAgent
from agents.td.double_q_learning_agent import DoubleQLearningAgent
from agents.td.expected_sarsa_agent import ExpectedSarsaAgent
from agents.td.q_learning_agent import QLearningAgent
from agents.td.sarsa_agent import SarsaAgent
from agents.wappers.state_wrapper_agent import StateWrapperAgent
from demo.sab.blackjack.blackjack_state_flattener import BlackjackStateFlattener, N_DEALER_STATES, N_PLAYER_STATES, \
    N_ACE_STATES
from model.agent_training_config import AgentTrainingConfig
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
    n_episodes = 200_000
    cfg = AgentTrainingConfig(epsilon=0.3, epsilon_decay=0.3/(n_episodes/2), min_epsilon=0, gamma=1.0, alpha=0.01)
    agent = OffPolicyNStepQSigmaAgent(N_DEALER_STATES * N_PLAYER_STATES * N_ACE_STATES, 2, cfg, n_step_size=2)
    flattener = BlackjackStateFlattener()
    agentw = StateWrapperAgent(agent, flattener)

    plotter = BlackjackStatePlotter("Blackjack-Q(0)-N2", 2, 1)

    train(env, agentw, n_episodes)
    state_grid = agentw.state_values_max()
    policy_grid = agentw.get_policy()

    plotter.create_plot(state_grid[:, :, 1], policy_grid[:, :, 1], f"{n_episodes} steps(Usable Ace)", 0)
    plotter.create_plot(state_grid[:, :, 0], policy_grid[:, :, 0], f"{n_episodes} steps(No Usable Ace)", 1)

    print(flattener.coverage)
    print(flattener.ace_sums)

    #plotter.show()
    plotter.save("blackjack-Q0N2-200000")


if __name__ == '__main__':
    start()
