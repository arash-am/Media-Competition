import numpy as np
import matplotlib.pyplot as plt


def plot_and_log(epoch,X, C, Loss, env, reward_total, loss_p, loss_n, cmap):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5));
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    # Plot collected opinions vs. susceptibility
    if len(X) > 0:
        # X is a list of final opinion-arrays, shape (N,) each episode
        X_np = np.array(X)  # shape (num_episodes, N)
        # Flatten them all to match with repeated susceptibility
        x_concat = X_np.reshape(-1)  # shape (num_episodes*N,)
        # If env.s is shape (N,), we tile it to match x_concat length
        s = env.s.detach().cpu().numpy()  # shape (N,)
        # Repeat for each episode in X
        s_concat = np.tile(s, X_np.shape[0])  # shape (num_episodes*N,)

        ax[0].scatter(x_concat, s_concat, c=x_concat, s=0.5, cmap=cmap, alpha=0.5)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(-1, 1)
    ax[0].set_xlabel("Opinion")
    ax[0].set_ylabel("Susceptibility")

    # Plot average credibility
    if len(C) > 0:
        # Each c_final is shape (M,)
        C_np = np.array(C)  # shape (num_episodes, M)
        c_mean = C_np.mean(axis=0)
        domain = np.linspace(-1, 1, env.M)
        ax[1].scatter(domain, c_mean, c=domain, cmap=cmap)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(-1, 1)
        ax[1].set_xlabel("Opinion-Bias")
        ax[1].set_ylabel("Credibility")

    # Plot losses (semilogy)
    if len(Loss) > 0:
        ax[2].semilogy(Loss)
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss (MSE)")


    plt.tight_layout()
    plt.show()

    # Print debugging info
    mean_loss = np.mean(Loss) if len(Loss) > 0 else 0.0
    mean_reward = np.mean(reward_total) if len(reward_total) > 0 else 0.0
    print(f"Epoch: {epoch} | Loss: {mean_loss:.3e} | Normal Loss: {loss_n:.3e} "
          f"| Reward: {mean_reward:.3e}")
    if len(C) > 0:
        print("Mean C:", C_np.mean(axis=0))