import os
import numpy as np
import torch
from tqdm import tqdm

from Files.equilibrium import *


# Ensure the directory exists
os.makedirs('Payoff_matrix_rationality', exist_ok=True)
# Parameters
M = 10
N = 500
terminal_time = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
taw_list = np.arange(-1, 2.1, 0.1)

env = OpinionEnv9Actions(
    num_envs=1000,
    N=N,
    M=M,
    terminal_time=terminal_time,
    bM=5,
    b=20,
    noise_level=0.1,
    h=0.1,
    nbins=30,
    r_scale=100,
    eta=1,
    eta2=1,
    beta_1=3,
    beta_2=2,
    lambda_c=0.98
)
# Initialize the payoff matrix A
A = torch.zeros((env.action_dim, env.action_dim), device=device)
# Populate A by running simulations for each pair of actions
for i in range(env.action_dim):
    for j in range(env.action_dim):
        A[i, j] = run_simulation(i, j, env)
# Save the payoff matrix to a file with a descriptive name
save_file = f'Payoff_matrix_rationality/Payoff.pt'
torch.save(A, save_file)
os.makedirs('results_rationality', exist_ok=True)
# Define the ranges for eta values.
data = {}       # Dictionary to store full simulation output
# Loop over the grid of eta values with nested progress bars.
for i, taw in tqdm(enumerate(taw_list), total=len(taw_list), desc="Rationality Loop"):
        # Run the simulation for the current eta1 and eta2 combination.
        env.reset()
        x, s, c, AEm = Generator(env,Taw=10**taw)
        # Store the results in the dictionary.
        data[i] = {'x': x,
                      's': s,
                      'c': c,
                      'AEm': AEm,
                      'bimodality':bimodality_coefficient(x),
                      'avg_c':c.mean(),
                      'avg_AE':AEm.mean() if AEm is not None else np.nan
                      }

torch.save(data, 'results_rationality/data-final.pt')

