from copy import copy

from agents.base_agent import BaseAgent
import numpy as np

from model.agent_training_config import AgentTrainingConfig


class OffPolicyMcAgent(BaseAgent):

    def __init__(self, n_states, n_actions, config: AgentTrainingConfig, b_of_s=None, b_of_a_given_s=None):
        super().__init__(config, n_actions, n_states, "OffPolicyMcAgent")
        self.n_states = n_states
        self.n_actions = n_actions

        self.transitions = []

        self.Q = np.zeros((n_states, n_actions))
        self.C = np.zeros((n_states, n_actions))

        self.pi_of_s = lambda s: self.greedy_action_select(s)
        self.pi_of_a_given_s = lambda a, s: 1 if self.greedy_action_select(s) == a else 0

        self.b_of_s = b_of_s
        if self.b_of_s is None:
            self.c.epsilon_decay = None  # epsilon should not decay, b is an exploratory policy
            self.b_of_s = lambda s: self.epsilon_greedy_action_select(s)

        self.b_of_a_given_s = b_of_a_given_s
        if self.b_of_a_given_s is None:
            self.b_of_a_given_s = lambda a, s: ((1 - self.c.epsilon) + (self.c.epsilon / self.n_actions)) \
                if self.greedy_action_select(s) == a else (self.c.epsilon / self.n_actions)

    def get_action(self, state):
        return self.b_of_s(state)

    def update(self, state, action, reward, terminated, next_state):
        self.transitions.append((state, action, reward, next_state))

        if terminated:
            self.do_episode_ended()

        super().update(state, action, reward, terminated, next_state)

    def do_episode_ended(self):
        self.do_reverse_transition_loop(self.transitions)
        self.transitions = []

    def do_reverse_transition_loop(self, transitions):
        G = 0
        W = 1
        for t in reversed(transitions):
            st, at, rt, ns = t

            G = self.c.gamma * G + rt

            # record training error
            self.add_training_error(G - self.Q[st, at])

            self.C[st, at] += W
            self.Q[st, at] += (W / self.C[st, at]) * (G - self.Q[st, at])

            if at != self.pi_of_s(st):
                break

            W *= 1 / self.b_of_a_given_s(at, st)

    def action_values(self):
        return self.Q
