import torch
import os

from Files.Env import Opinion_w_media
from Files.soft_nash import soft_q_net, replay_buffer, train
from Files.plotter import plot_and_log
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_blue_red", [(0, 0, 1), (1, 0, 0)], N=100)
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
nbins = 30

env = Opinion_w_media(N=N,
                      M=M,
                      terminal_time=401,
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
#  3) Hyperparameters
#########################################
gamma = 0.95
learning_rate = 1e-4
batch_size = 64
capacity = 3*10**5
episode = 2_000_000
observation_dim = nbins + M
bpl = 10
bop = -10
TAU = 0.002

# Number of possible discrete actions (for M=10, 2^(M//2))
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
epoch = 0
max_epochs = 40000
Done = False
C = []
X = []
Loss = []

# Ensure "models" directory exists
os.makedirs("models", exist_ok=True)

#########################################
#   Main Loop (Training)
#########################################
for i in range(episode):
    xct = env.reset()           # Reset environment
    obs = env.state2obs(xct)    # Observation
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

        # Accumulate reward
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
                    plot_and_log(epoch, X, C, Loss, env, reward_total, loss_p, loss_n,cmap)
                    reward_total = []
                    C = []
                    X = []
                    avg_net.load_state_dict(eval_net.state_dict())

                # ------------------------------
                # SAVE CHECKPOINT every 1000 epochs
                # ------------------------------
                if epoch % 1000 == 0 and epoch > 0:
                    ckpt_path = f"models/checkpoint_epoch_{epoch}.pth"
                    torch.save({
                        'epoch': epoch,
                        'eval_net_state_dict': eval_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, ckpt_path)
                    print(f"Checkpoint saved at {ckpt_path}")

        if Done:
            # On termination, grab final opinions & credibility
            x_final = n_xct[:env.N].detach().cpu().numpy()
            c_final = n_xct[env.N:-1].detach().cpu().numpy()
            X.append(x_final)
            C.append(c_final)
            break

print("Training complete.")

#########################################
#  6) Save final model
#########################################
final_model_path = "models/final_eval_net.pth"
torch.save(eval_net.state_dict(), final_model_path)
print(f"Final model saved as {final_model_path}")
