from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class FrozenLakePlotter:

    def __init__(self, Q, n_rows: int, n_cols: int, title: str = None):
        self.title = title

        state_value = defaultdict(float)
        policy = defaultdict(int)

        for obs, action_values in enumerate(Q):
            state_value[obs] = float(np.max(action_values))
            policy[obs] = int(np.argmax(action_values))

        xx, yy = np.meshgrid(
            np.arange(0, n_cols),
            np.arange(0, n_rows),
        )

        # create the value grid for plotting
        self.value_grid = np.apply_along_axis(
            lambda obs: state_value[self.xy_to_cell_index(obs[0], obs[1], n_cols)],
            axis=2,
            arr=np.dstack([xx, yy]),
        )

        # create the policy grid for plotting
        self.policy_grid = np.apply_along_axis(
            lambda obs: policy[self.xy_to_cell_index(obs[0], obs[1], n_cols)],
            axis=2,
            arr=np.dstack([xx, yy]),
        )

        # create the policy grid for plotting
        u = np.apply_along_axis(
            lambda obs: self.u_map(policy[self.xy_to_cell_index(obs[0], obs[1], n_cols)]),
            axis=2,
            arr=np.dstack([xx, yy]),
        )

        v = np.apply_along_axis(
            lambda obs: self.v_map(policy[self.xy_to_cell_index(obs[0], obs[1], n_cols)]),
            axis=2,
            arr=np.dstack([xx, yy]),
        )

        self.arrow_grid = (xx, yy, u, v)

    def xy_to_cell_index(self, x: int, y: int, n_cols: int):
        return y * n_cols + x

    def cell_index_to_xy(self, cell_index: int, n_cols: int):
        return cell_index % n_cols, cell_index // n_cols

    def u_map(self, val):
        return 0 if val == 3 else (val - 1)

    def v_map(self, val):
        return 0 if val == 0 else (val - 2)

    def show(self):
        fig = self.create_plots()
        fig.show()

    def create_plots(self):
        """Creates a plot using a value and policy grid."""

        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(self.title, fontsize=16)

        # plot the policy
        fig.add_subplot(1, 3, 1)
        ax1 = sns.heatmap(self.value_grid, linewidth=0, annot=True, cmap="cool", cbar=False)
        ax1.set_title(f"State values")
        ax1.set_xlabel("Cols")
        ax1.set_ylabel("Rows")
        ax1.set_xticklabels(range(0, 4))
        ax1.set_yticklabels(range(0, 4))

        # plot the policy
        fig.add_subplot(1, 3, 2)
        ax2 = sns.heatmap(self.policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
        ax2.set_title(f"Policy")
        ax2.set_xlabel("Cols")
        ax2.set_ylabel("Rows")
        ax2.set_xticklabels(range(0, 4))
        ax2.set_yticklabels(range(0, 4))

        xx, yy, u, v = self.arrow_grid
        ax2 = fig.add_subplot(1, 3, 3)
        ax2.set_title(f"Arrows")
        ax2.quiver(xx, list(reversed(yy)), u, v)
        ax2.xaxis.set_ticks([])
        ax2.yaxis.set_ticks([])
        ax2.set_aspect('equal')

        ax2.legend()

        return fig
