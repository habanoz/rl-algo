import numpy as np
from numpy import ndarray

from agents.wrappers.state_coder import StateFlattener

MIN_PLAYER_SUM = 12
N_PLAYER_STATES = 10
MIN_DEALER_CARD = 1
N_DEALER_STATES = 10
N_ACE_STATES = 2
N_ACTIONS = 2


class BlackjackStateFlattener(StateFlattener):
    def __init__(self):
        self.coverage = set()
        self.ace_sums = set()

    def flatten(self, obs):
        aligned_obs = self._zero_align(obs)
        return self._flatten_zero_aligned(aligned_obs)

    def _flatten_zero_aligned(self, obs):
        player_sum, dealer_showing, usable_ace = obs
        s = (player_sum * N_DEALER_STATES + dealer_showing) * N_ACE_STATES + usable_ace

        self.coverage.add(s)

        return s

    def _zero_align(self, obs):
        player_sum, dealer_showing, usable_ace = obs

        if int(usable_ace) == 1:
            self.ace_sums.add(player_sum)

        player_sum = player_sum - MIN_PLAYER_SUM

        assert player_sum >= 0

        if player_sum > N_PLAYER_STATES:
            player_sum = N_PLAYER_STATES

        return player_sum, dealer_showing - MIN_DEALER_CARD, int(usable_ace)

    def deflatten_state_values(self, flat_states: ndarray):
        value_grid = np.empty((N_PLAYER_STATES, N_DEALER_STATES, N_ACE_STATES))

        for p in range(N_PLAYER_STATES):
            for d in range(N_DEALER_STATES):
                for b in range(N_ACE_STATES):
                    value_grid[p, d, b] = flat_states[self._flatten_zero_aligned((p, d, b))]

        return value_grid

    def deflatten_action_values(self, flat_action_values):
        value_grid = np.empty((N_PLAYER_STATES, N_DEALER_STATES, N_ACE_STATES, N_ACTIONS))

        for p in range(N_PLAYER_STATES):
            for d in range(N_DEALER_STATES):
                for b in range(N_ACE_STATES):
                    for a in range(N_ACTIONS):
                        value_grid[p, d, b, a] = flat_action_values[self._flatten_zero_aligned((p, d, b)), a]

        return value_grid

    def deflatten_policy(self, flat_action_values):
        policy_grid = np.empty((N_PLAYER_STATES, N_DEALER_STATES, N_ACE_STATES))

        for p in range(N_PLAYER_STATES):
            for d in range(N_DEALER_STATES):
                for b in range(N_ACE_STATES):
                    policy_grid[p, d, b] = np.argmax(flat_action_values[self._flatten_zero_aligned((p, d, b)), :])

        return policy_grid
