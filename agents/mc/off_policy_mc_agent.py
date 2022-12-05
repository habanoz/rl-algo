from agents.base_agent import BaseAgent
import numpy as np

from transition import Transition


class OffPolicyMcAgent(BaseAgent):
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.9):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gama = gamma
        self.transitions = []

        self.Q = np.zeros((n_states, n_actions))
        self.C = np.zeros((n_states, n_actions))

        self.b = np.full((n_states, n_actions), max(self.epsilon / n_actions, 1 / n_actions))  # epsilon soft-policy
        self.pi = np.full(n_states, 0)

    def get_action(self, state):
        return np.random.choice(np.arange(self.n_actions), p=self.b[state, :])

    def update(self, state, action, reward, terminated, next_state):
        self.transitions.append(Transition(state, action, reward, next_state))

        if terminated:
            self.do_episode_ended()

    def do_episode_ended(self):
        self.do_reverse_transition_loop(self.transitions)
        self.transitions = []

    def do_reverse_transition_loop(self, transitions):
        G = 0
        W = 1
        for t in reversed(transitions):
            (st, at, rt, ns, first_visit) = t.to_tuple()

            G = self.gama * G + rt

            self.C[st, at] += W
            self.Q[st, at] += (W / self.C[st, at])*(G - self.Q[st, at])

            self.pi[st] = self.greedy_action_select(self.Q[st, :])

            if at != self.pi[st]:
                break

            W *= 1 / self.b[st, at]
