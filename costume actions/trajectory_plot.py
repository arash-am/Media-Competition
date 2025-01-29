import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def generate_and_save_plots(env, states, output_folder="plots", plt_type='Normal'):
    """
    Generates and saves plots based on the environment and state data.

    Parameters:
    env : object
        The environment containing simulation data.
    states : np.ndarray
        The state data to visualize.
    output_folder : str, optional
        The folder where plots will be saved (default is "plots").
    plt_type : str, optional
        Plot type identifier (default is "Normal").
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Opinion History Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    X = states[:env.N, 0:env.terminal_time+1].numpy()
    T = np.ones((1, env.N)) * np.arange(X.shape[1])[:, None]

    cax = sns.histplot(x=T.T.flatten(), y=X.flatten(), fill=True, cmap="RdYlBu_r", bins=100, thresh=None, ax=ax)
    quadmesh = cax.collections[0]
    quadmesh.set_clim(0, 40)
    cbar = plt.colorbar(cax.collections[0], ax=ax, aspect=25)
    cbar.ax.tick_params(labelsize=15, width=2, length=10)
    cbar.ax.set_yticks([0, 10, 20, 30, 40])

    ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
    ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, env.terminal_time-1)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([0, 100, 200])
    ax.set_xlabel("Time", fontsize=20)
    ax.set_ylabel("Opinion", fontsize=20)
    ax.set_title("Opinion History", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"op_hist_{plt_type}.svg"), dpi=300)
    plt.savefig(os.path.join(output_folder, f"op_hist_{plt_type}.png"), dpi=100)
    plt.close()

    # KDE Plot for Different Time Indices
    time_indices = [0, 100, 200,300,400]
    for t_idx in time_indices:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.kdeplot(x=X[:, t_idx], y=env.s.squeeze().cpu().numpy(), fill=True, thresh=None, cmap="magma", levels=100, ax=ax)

        ax.set_xlim([-1.15, 1.15])
        ax.set_ylim(0, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
        ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_title(f"Time = {t_idx}", fontsize=20)
        ax.set_xlabel("Opinion", fontsize=20)
        ax.set_ylabel("Susceptibility", fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"xs_kde{t_idx:03}_{plt_type}.svg"), dpi=300)
        plt.close()

    # Credibility Profile
    cmap = LinearSegmentedColormap.from_list("custom_blue_red", [(0, 0, 1), (1, 0, 0)], N=100)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.scatter(env.ym.cpu(), states[env.N:, -1].numpy(), 600, c=env.ym.cpu(), cmap=cmap, alpha=1)

    ax.set_xlim([-1.15, 1.15])
    ax.set_ylim([-0.15, 1.15])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
    ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_title("Credibility Profile", fontsize=20)
    ax.set_xlabel("Opinion", fontsize=20)
    ax.set_ylabel("Credibility", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"cx_circ_{plt_type}.svg"), dpi=300)
    plt.close()

    # Misinformation Exposure Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y=env.AEm_abs.sum(axis=1).cpu().numpy() / (env.terminal_time-1), x=X[:, -1], c=X[:, -1], cmap=cmap, s=100, alpha=1)

    ax.set_xlim([-1.15, 1.15])
    ax.set_ylim(0, 1.5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
    ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel("Opinion", fontsize=20)
    ax.set_ylabel("Misinformation Exposure", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Ex_{plt_type}.svg"), dpi=300)
    plt.close()

    print("Finished generating and saving plots.", "*" * 50)
