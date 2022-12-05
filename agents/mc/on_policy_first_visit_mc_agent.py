from collections import defaultdict

from agents.base_agent import BaseAgent
from transition import Transition
import numpy as np


class OnPolicyFirstVisitMcAgent(BaseAgent):
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.9):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gama = gamma
        self.transitions = []
        self.returns = defaultdict(list)
        self.visits = set()
        self.Q = np.zeros((n_states, n_actions))
        self.pi = np.full((n_states, n_actions), max(self.epsilon / n_actions, 1 / n_actions))  # epsilon soft-policy

    def get_action(self, state):
        return np.random.choice(np.arange(self.n_actions), p=self.pi[state, :])

    def update(self, state, action, reward, terminated, next_state):
        first_visit = (state, action) not in self.visits
        if not first_visit:
            self.visits.add((state, action))

        self.transitions.append(Transition(state, action, reward, next_state, first_visit))

        if terminated:
            self.do_episode_ended()

    def do_episode_ended(self):
        self.do_reverse_transition_loop(self.transitions)
        self.visits = set()
        self.transitions = []

    def do_reverse_transition_loop(self, transitions):
        G = 0
        for t in reversed(transitions):
            (st, at, rt, ns, first_visit) = t.to_tuple()

            G = self.gama * G + rt

            if first_visit:
                self.returns[(st, at)].append(G)
                self.Q[st, at] = np.mean(self.returns[(st, at)])

                a_star = self.arg_max(self.Q[st, :])

                for a in range(self.n_actions):
                    if a == a_star:
                        self.pi[st, a] = 1 - self.epsilon + self.epsilon / self.n_actions
                    else:
                        self.pi[st, a] = self.epsilon / self.n_actions
