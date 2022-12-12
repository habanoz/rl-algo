import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

from demo.sab.blackjack.blackjack_state_flattener import MIN_PLAYER_SUM, MIN_DEALER_CARD


class BlackjackStatePlotter:

    def __init__(self, title: str, n_rows, n_cols):
        self.n_cols = n_cols
        self.n_rows = n_rows

        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.fig.suptitle(title, fontsize=16)

    def _create_grid(self, state_values_array2d, policy_array2d):
        player_count, dealer_count = np.meshgrid(
            np.arange(12, 22),
            np.arange(1, 11),
        )

        value = np.apply_along_axis(
            lambda obs: state_values_array2d[obs[0] - MIN_PLAYER_SUM, obs[1] - MIN_DEALER_CARD],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )

        value_grid = player_count, dealer_count, value

        policy_grid = np.apply_along_axis(
            lambda obs: policy_array2d[obs[0] - MIN_PLAYER_SUM, obs[1] - MIN_DEALER_CARD],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )

        return value_grid, policy_grid

    def show(self):
        plt.tight_layout()
        plt.show()

    def create_plot(self, state_values_array2d: ndarray, policy_array2d: ndarray, title: str, idx=0):
        """Creates a plot using a value and policy grid."""

        value_grid, policy_grid = self._create_grid(state_values_array2d, policy_array2d)
        player_count, dealer_count, value = value_grid

        ax1 = self.fig.add_subplot(self.n_rows, self.n_cols * 2, 2 * idx + 1, projection="3d")
        surf = ax1.plot_surface(
            player_count,
            dealer_count,
            value,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none"
        )
        plt.xticks(range(12, 22), range(12, 22))
        plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
        ax1.set_zticks([-1, 1])
        ax1.set_title(f"State values: {title}")
        ax1.set_xlabel("Player sum")
        ax1.set_ylabel("Dealer showing")
        ax1.zaxis.set_rotate_label(False)
        ax1.set_zlabel("Value", fontsize=14, rotation=90)

        scale = np.diag([1.0, 1.0, 0.5, 1.0])
        scale = scale * (1.0 / scale.max())
        scale[3, 3] = 1.0

        def short_proj():
            return np.dot(Axes3D.get_proj(ax1), scale)

        ax1.get_proj = short_proj

        ax1.invert_yaxis()

        ax1.view_init(45, 220)
        ax1.legend()

        self.fig.colorbar(surf, shrink=0.5, aspect=5)

        # plot the policy
        self.fig.add_subplot(self.n_rows, self.n_cols * 2, 2 * idx + 2)
        ax2 = sns.heatmap(np.transpose(policy_grid), linewidth=0, annot=True, cmap="Accent_r", cbar=False)
        ax2.set_title(f"Policy: {title}")
        ax2.set_ylabel("Player sum")
        ax2.set_xlabel("Dealer showing")
        ax2.set_yticklabels(range(12, 22))
        ax2.set_xticklabels(["A"] + list(range(2, 11)), fontsize=12)

        ax2.invert_yaxis()

        # add a legend
        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
            Patch(facecolor="grey", edgecolor="black", label="Stick"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
