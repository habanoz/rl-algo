from matplotlib.axes import Axes


class StateValuePlotter:
    def __init__(self, state_value_grid):
        self.state_value_grid = state_value_grid

    def plot(self, axis:Axes):
        axis.plot()