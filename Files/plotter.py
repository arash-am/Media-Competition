import numpy as np
import matplotlib.pyplot as plt


def console_bar_plot(values, label="Credibility"):
    """Generates an ASCII bar plot string."""
    max_width = 50  # Max width for the bars
    max_val = max(values) if len(values) > 0 else 1.0  # Normalize based on max value

    output = f"\n{label}:\n"
    for i, val in enumerate(values):
        bar_length = int((val / max_val) * max_width)  # Scale the bar length
        bar = "#" * bar_length
        output += f"[{i:2d}] {bar} ({val:.2f})\n"

    return output


def plot_and_log(epoch, X, C, Loss, env, reward_total, loss_p, loss_n, cmap, log_interval=100, plot_interval=1000):
    """Logs metrics every log_interval and updates graphical plots every plot_interval."""

    # Compute statistics
    mean_loss = np.mean(Loss) if len(Loss) > 0 else 0.0
    mean_reward = np.mean(reward_total) if len(reward_total) > 0 else 0.0

    # Print logs every `log_interval` epochs
    if epoch % log_interval == 0:
        log_text = f"Epoch: {epoch}\nLoss: {mean_loss:.3e} | Normal Loss: {loss_n:.3e} | Reward: {mean_reward:.3e}"
        print(log_text)

        # Print credibility bar plot if data exists
        if len(C) > 0:
            C_np = np.array(C)
            c_mean = C_np.mean(axis=0)
            print(console_bar_plot(c_mean, label="Credibility"))

    # Update graphical plots every `plot_interval` epochs
    if epoch % plot_interval == 0:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot collected opinions vs. susceptibility
        if len(X) > 0:
            X_np = np.array(X)
            x_concat = X_np.reshape(-1)
            s = env.s.detach().cpu().numpy()
            s_concat = np.tile(s, X_np.shape[0])

            ax[0].scatter(x_concat, s_concat, c=x_concat, s=0.5, cmap=cmap, alpha=0.5)
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(-1, 1)
        ax[0].set_xlabel("Opinion")
        ax[0].set_ylabel("Susceptibility")

        # Plot average credibility
        if len(C) > 0:
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

