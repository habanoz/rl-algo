from collections import defaultdict

from agents.base_agent import BaseAgent
from model.agent_config import AgentConfig
from transition import Transition
import numpy as np


class OnPolicyFirstVisitMcAgent(BaseAgent):
    def __init__(self, n_states, n_actions, config: AgentConfig):
        super().__init__(config)
        self.n_states = n_states
        self.n_actions = n_actions

        self.transitions = []
        self.returns = defaultdict(list)
        self.visits = set()
        self.Q = np.zeros((n_states, n_actions))
        self.pi = np.full((n_states, n_actions), max(self.c.epsilon / n_actions, 1 / n_actions))  # epsilon soft-policy

    def get_action(self, state):
        return np.random.choice(np.arange(self.n_actions), p=self.pi[state, :])

    def update(self, state, action, reward, terminated, next_state):
        first_visit = (state, action) not in self.visits
        if not first_visit:
            self.visits.add((state, action))

        self.transitions.append(Transition(state, action, reward, next_state, first_visit))

        if terminated:
            self.do_episode_ended()

        super().update(state, action, reward, terminated, next_state)

    def do_episode_ended(self):
        self.do_reverse_transition_loop(self.transitions)
        self.visits = set()
        self.transitions = []

    def do_reverse_transition_loop(self, transitions):
        G = 0
        for t in reversed(transitions):
            (st, at, rt, ns, first_visit) = t.to_tuple()

            G = self.c.gamma * G + rt

            if first_visit:
                # record training error
                if len(self.returns[(st, at)]) > 0:
                    self.add_training_error(G, np.mean(self.returns[(st, at)]))

                self.returns[(st, at)].append(G)
                self.Q[st, at] = np.mean(self.returns[(st, at)])

                a_star = self.greedy_action_select(self.Q[st, :])

                for a in range(self.n_actions):
                    if a == a_star:
                        self.pi[st, a] = 1 - self.c.epsilon + self.c.epsilon / self.n_actions
                    else:
                        self.pi[st, a] = self.c.epsilon / self.n_actions

    def state_values(self):
        return np.array([np.mean(r) for r in self.Q])
