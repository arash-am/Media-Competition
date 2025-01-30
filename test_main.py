import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.stats as stats

from Files.Env import Opinion_w_media
from Files.soft_nash import soft_q_net

#########################################
#  1) Setup the device
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#########################################
#  2) Create the same environment
#########################################
N = 500
M = 10
terminal_time = 200
nbins = 30

env = Opinion_w_media(N=N,
                      M=M,
                      terminal_time=201,
                      bM=4,
                      b=18,
                      noise_level=0.1,
                      duration=1,
                      h=torch.tensor(0.1, device=device),
                      nbins=30,
                      r_scale=100,
                      eta_1=1,
                      eta_2=1,
                      beta_1=3,
                      beta_2=2,
                      )

#########################################
#  3) Define the same network structure
#########################################
gamma = 0.98
learning_rate = 1e-3
batch_size = 64
capacity = 5*10**5
episode = 10_000_000
observation_dim = nbins + M
bpl = 20
bop = -20
TAU = 0.003
action_dimension = 2 ** (M // 2)  # must match training

test_net = soft_q_net(observation_dim, bpl, bop, M, action_dimension).to(device)

#########################################
#  4) LOAD the final trained model weights
#########################################
model_path = "models/final_eval_net.pth"  # or use "checkpoint_epoch_1000.pth" etc.
# model_path = "models/checkpoint_net.pth"
test_net.load_state_dict(torch.load(model_path, map_location=device))
test_net.eval()
print(f"Loaded model from {model_path}")

#########################################
#  5) Run the testing (K episodes)
#########################################
K = 100
c_mean = torch.zeros(M, device=device)
x_mean = []
count = 0
R_mean = 0

# For storing the entire trajectory across all rollouts:
X_test = torch.zeros((N, env.terminal_time, K), device=device)
C_test = torch.zeros((M, env.terminal_time, K), device=device)

with torch.no_grad():
    for j in range(K):
        xct = env.reset()
        obs = env.state2obs(xct)
        reward_total = 0.0

        while True:
            u_action, v_action, action_id, _ = test_net.act(obs.unsqueeze(0).to(device))

            n_xct, reward, Done, _ = env.step(u_action, v_action)
            next_obs = env.state2obs(n_xct)

            # Track data
            x, c, t = n_xct[:N], n_xct[N:-1], n_xct[-1]
            reward_total += reward / env.terminal_time

            # Zero-based time index
            t_int = int(t.item()) - 1
            if 0 <= t_int < env.terminal_time:
                X_test[:, t_int, j] = x
                C_test[:, t_int, j] = c

            obs = next_obs
            if Done:
                count += 1
                c_mean += c / K
                x_mean.append(x)
                R_mean += reward_total / K
                break

print(f"Test episodes finished = {count}")
print(f"Mean reward across {K} episodes = {R_mean:.3f}")

#########################################
#  6) Plotting Equilibrium Distributions
#########################################

# Example: Plot final opinions vs. s
x_all = X_test[:, -1, :].reshape(-1).cpu().numpy()
s_all = env.s.squeeze().repeat(K).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.kdeplot(x=x_all, y=s_all, fill=True, thresh=None, cmap="magma", levels=100, ax=ax)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(0, 1)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 1])
ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tight_layout()
plt.savefig('xs200-complete_action_b10.svg')
plt.show()

# Plot final average credibility
cmap = LinearSegmentedColormap.from_list("custom_blue_red", [(0, 0, 1), (1, 0, 0)], N=100)
fig, ax = plt.subplots(figsize=(5, 2.5))
ym_cpu = env.ym.cpu().numpy()
c_mean_cpu = c_mean.cpu().numpy()
ax.scatter(ym_cpu, c_mean_cpu, s=600, c=ym_cpu, cmap=cmap, alpha=1)
ax.set_xlim([-1.15, 1.15])
ax.set_ylim([-0.15, 1.15])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 1])
ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tight_layout()
plt.savefig('xc200-complete_action_10.svg')
plt.show()

# Plot opinion distribution over time (like a ridge plot)
fig, ax = plt.subplots(figsize=(6, 8))
cmap2 = plt.get_cmap("viridis")
colors = [cmap2(i) for i in np.linspace(0, 1, 20)]

vertical_shift = 0.18
scale_factor = 0.35
yp = 0.0

time_indices_to_plot = range(0, env.terminal_time, 20)
for idx, t in enumerate(time_indices_to_plot):
    data = X_test[:, t, :].reshape(-1).cpu().numpy()
    kde = stats.gaussian_kde(data)
    x_vals = np.linspace(-1.3, 1.3, 1000)
    y_vals = kde(x_vals) * scale_factor + idx * vertical_shift
    ax.fill_between(
        x_vals,
        np.maximum(idx * vertical_shift, yp),
        y_vals,
        color=colors[idx],
        alpha=8/(idx+10)
    )
    ax.plot(x_vals, y_vals, color='white', linewidth=3)
    yp = y_vals

ax.set_xlim([-1.3, 1.3])
ax.set_ylim([0, idx * vertical_shift + scale_factor])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 3*vertical_shift, 6*vertical_shift, 9*vertical_shift])
ax.tick_params(axis='x', which='major', labelsize=15, width=2, length=10)
ax.tick_params(axis='y', which='major', labelsize=15, width=2, length=10)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tight_layout()
plt.savefig('xt200-complete_action_b10.svg')
plt.show()
