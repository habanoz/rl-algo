from numpy.core.records import ndarray
import numpy as np


class RMSErrorsForBaseline:
    def __init__(self, value_baseline):
        self.value_baseline = value_baseline

    def calculate_for_state(self, state_values: ndarray):
        if self.value_baseline is None:
            return 0

        assert len(state_values) == len(self.value_baseline)
        return np.sqrt(np.sum(np.power(state_values - self.value_baseline, 2)) / len(state_values))

    def calculate_for_action(self, action_values: ndarray):
        return self.calculate_for_state(np.array([np.mean(r) for r in action_values]))
