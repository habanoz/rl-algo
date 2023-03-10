from collections import defaultdict

import numpy as np

from agents.base_agent import BaseAgent, AgentTrainingConfig


class OnPolicyFirstVisitMcAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig):
        super().__init__(config, n_actions, n_states, "OnPolicyFirstVisitMcAgent")

        self.transitions = []
        self.return_sums = defaultdict(float)
        self.return_counts = defaultdict(lambda: 1)
        self.visits = set()
        self.Q = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, terminated, next_state, truncated=False):
        first_visit = (state, action) not in self.visits
        if not first_visit:
            self.visits.add((state, action))

        self.transitions.append((state, action, reward, next_state, first_visit))

        if terminated:
            self.do_episode_ended()

        super().update(state, action, reward, terminated, next_state)

    def do_episode_ended(self):
        self.do_reverse_transition_loop(self.transitions)
        self.do_after_episode()
        self.visits = set()
        self.transitions = []

    def do_reverse_transition_loop(self, transitions):
        G = 0
        for t in reversed(transitions):
            st, at, rt, ns, first_visit = t

            G = self.c.gamma * G + rt

            if first_visit:
                # record training error
                self.add_training_error(G - self.return_sums[(st, at)] / self.return_counts[(st, at)])

                self.return_sums[(st, at)] = self.return_sums[(st, at)] + G
                self.return_counts[(st, at)] = self.return_counts[(st, at)] + 1

                self.Q[st, at] = self.return_sums[(st, at)] / self.return_counts[(st, at)]

    def action_values(self):
        return self.Q
