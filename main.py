from Files.Env import *
from Files.soft_nash import *
from Files.plotter import *
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

#########################################
#  1) Setup the device for CUDA
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#########################################
#  2) Create Environment
#########################################
N = 500
M = 10
terminal_time = 200
Duration = 1
nbins = 30
scale = 100

env = Opinion_w_media(N=N,
                      M=M,
                      terminal_time=terminal_time,
                      bM=5,
                      b=20,
                      noise_level=0.1,
                      duration=Duration,
                      h=torch.tensor(0.1, device=device),
                      nbins=nbins,
                      r_scale=scale)

#########################################
#  3) Hyperparameters
#########################################
gamma = 0.95
learning_rate = 1e-4
batch_size = 64
capacity = 10**5
episode = 2000000
observation_dim = nbins + M
bpl = 10
bop = -10
TAU = 0.002
action_dimension = 2 ** (M // 2)

#########################################
#  4) Networks
#########################################
target_net = soft_q_net(observation_dim, bpl, bop, M, action_dimension).to(device)
eval_net   = soft_q_net(observation_dim, bpl, bop, M, action_dimension).to(device)
avg_net    = soft_q_net(observation_dim, bpl, bop, M, action_dimension).to(device)

eval_net.load_state_dict(target_net.state_dict())
avg_net.load_state_dict(eval_net.state_dict())

optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate, weight_decay=1e-3)
buffer = replay_buffer(capacity)

#########################################
#  5) Training Helpers
#########################################
count = 0
reward_total = []
Loss = []
R_t = 0
loss_p = -1
epoch = 0
max_epochs = 40000

# For real-time plotting
# plt.ion()
cmap = LinearSegmentedColormap.from_list("custom_blue_red", [(0, 0, 1), (1, 0, 0)], N=100)

Done = False
C = []
X = []
Loss = []



#########################################
#   Main Loop                           #
#########################################
for i in range(episode):
    xct = env.reset()           # reset environment
    obs = env.state2obs(xct)    # observation
    if epoch > max_epochs:
        break

    while True:
        # Pick action
        u_action, v_action, action_id, action_dist = eval_net.act(obs.unsqueeze(0).to(device))
        count += 1

        # Step environment
        n_xct, reward, Done, _ = env.step(u_action, v_action)
        next_obs = env.state2obs(n_xct)

        # Store in replay
        buffer.store(obs.cpu(), action_id, reward.item(), next_obs.cpu(), float(Done))

        # Accumulate reward in a list for logging
        reward_total.append(reward.item())
        obs = next_obs

        # Train if we have enough data in buffer
        if len(buffer.memory) > batch_size:
            if count % 50 == 0:
                epoch += 1
                loss_p, loss_n, q_print = train(
                    buffer, target_net, eval_net, gamma, optimizer,
                    batch_size, count, None, TAU
                )
                Loss.append(loss_p)

                # Update avg_net
                for key in eval_net.state_dict():
                    avg_net.state_dict()[key] += eval_net.state_dict()[key] / 100.0

                # Occasionally plot
                if epoch % 100 == 0:
                    plot_and_log(epoch, X, C, Loss, env, reward_total, loss_p, loss_n, cmap)
                    # Reset these accumulators AFTER plotting
                    reward_total = []
                    C = []
                    X = []
                    avg_net.load_state_dict(eval_net.state_dict())

        if Done:
            # On termination, grab final opinions & credibility
            x_final = n_xct[:env.N].detach().cpu().numpy()      # shape (N,)
            c_final = n_xct[env.N:-1].detach().cpu().numpy()    # shape (M,)
            X.append(x_final)
            C.append(c_final)
            break

# Final plot after everything
plt.ioff()
plt.show()
print("Training complete.")
