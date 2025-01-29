from Files.Env import *
from trajectory_plot import *
import torch


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
#          Dynamic Loop                 #
#########################################
# setting up the actions

states=torch.zeros([M+N,env.terminal_time+1],device=device)

xct = env.reset()           # reset environment
states[:,0]=env.state[:-1]
done=False
while not done:
    # action selection
    actions = torch.bernoulli(.9 - 0.8*1e-0 * env.ym ** 2).to(device)[None, :]
    u_action = env.action_decoder(actions.numpy()[0][:int(M / 2)])
    v_action = env.action_decoder(actions.numpy()[0][int(M / 2):])
    # Step environment
    n_xct, reward, done, _ = env.step(u_action, v_action)
    states[:, int(env.state[-1])] = env.state[:-1]

generate_and_save_plots(env, states, output_folder="plots_real", plt_type='real')

