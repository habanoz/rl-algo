import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

from demo.sab.blackjack.blackjack_state_flattener import MIN_PLAYER_SUM, MIN_DEALER_CARD


class BlackjackStatePlotterCopied:

    def __init__(self, title: str, n_rows, n_cols):
        self.n_cols = n_cols
        self.n_rows = n_rows
        # self.fig, self.axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=size)

        self.fig = plt.figure(figsize=plt.figaspect(0.4))
        self.fig.suptitle(title, fontsize=16)

    def _create_grid(self, state_value, policy, usable_ace=False):
        player_count, dealer_count = np.meshgrid(
            # players count, dealers face-up card
            np.arange(12, 22),
            np.arange(1, 11),
        )

        # create the value grid for plotting
        value = np.apply_along_axis(
            lambda obs: state_value[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        value_grid = player_count, dealer_count, value

        # create the policy grid for plotting
        policy_grid = np.apply_along_axis(
            lambda obs: policy[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        return value_grid, policy_grid

    def show(self):
        plt.tight_layout()
        plt.show()

    def create_plot(self, state_values_array2d: ndarray, policy_array2d:ndarray, title: str, idx=0, usable_ace=False):
        """Creates a plot using a value and policy grid."""

        value_grid, policy_grid = self._create_grid(state_values_array2d, policy_array2d, usable_ace=usable_ace)
        player_count, dealer_count, value = value_grid

        ax1 = self.fig.add_subplot(2, 4, 2*idx + 1, projection="3d")
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

        # Add a color bar which maps values to colors.
        self.fig.colorbar(surf, shrink=0.5, aspect=5)

        # plot the policy
        self.fig.add_subplot(2, 4, 2*idx + 2)
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

        # # ax = self.axs.flat[idx]
        # ax = self.fig.add_subplot(self.n_rows, self.n_cols, idx + 1, projection="3d")
        # # ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        # ax.plot_surface(
        #     player_count,
        #     dealer_count,
        #     value,
        #     rstride=1,
        #     cstride=1,
        #     cmap="viridis",
        #     edgecolor="none",
        # )
        # # plt.yticks(range(12, 22), range(12, 22))
        # # plt.xticks(range(1, 11), ["A"] + list(range(2, 11)))
        # ax.set_title(f"State values: {title}")
        # ax.set_ylabel("Player sum")
        # ax.set_xlabel("Dealer showing")
        # ax.zaxis.set_rotate_label(False)
        # ax.set_zlabel("Value", fontsize=14, rotation=90)
        # ax.view_init(20, 220)

        #
        # # plot the policy
        # # self.fig.add_subplot((self.n_rows, self.n_cols, pos))
        # ax = self.axs.flat[idx]
        # ax = sns.heatmap(grid, linewidth=0, annot=True, cmap="cool", ax=ax, cbar=False)
        # ax.set_title(title)
        # ax.set_xlabel("Dealer showing")
        # ax.set_ylabel("Player sum")
        # ax.set_xticklabels(["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        # ax.set_yticklabels(["12", "13", "14", "15", "16", "17", "18", "19", "20", "21"])

        # ax.legend()
