import os
import numpy as np
import torch
from tqdm import tqdm

from Files.equilibrium import *


# Ensure the directory exists
os.makedirs('Payoff_matrix', exist_ok=True)
# Parameters
M = 10
N = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
terminal_time = 200
Eta1_list = np.arange(0, 2.1, 0.2)
Eta2_list = np.arange(0, 6.1, 0.2)

# Loop over eta1 and eta2 values with progress bars
for eta1 in tqdm(Eta1_list, desc='Outer Loop Progress'):
    for eta2 in tqdm(Eta2_list, desc=f'Inner Loop (eta1={eta1:.1f})', leave=False):
        # Instantiate the environment with the current eta1 and eta2 values
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
            eta=eta1,
            eta2=eta2,
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
        save_file = f'Payoff_matrix/Payoff-eta1_{round(eta1, 1)}eta2_{round(eta2, 1)}.pt'
        torch.save(A, save_file)



os.makedirs('results', exist_ok=True)
# Define the ranges for eta values.
data = {}       # Dictionary to store full simulation output
# Loop over the grid of eta values with nested progress bars.

for i, eta1 in tqdm(enumerate(Eta1_list), total=len(Eta1_list), desc="Outer Loop (Eta1)"):
    data[i] = {}
    for j, eta2 in tqdm(enumerate(Eta2_list), total=len(Eta2_list), desc=f"Inner Loop (eta1={eta1:.1f})", leave=False):
        # Run the simulation for the current eta1 and eta2 combination.
        env.eta= eta1
        env.eta2 = eta2
        env.reset()
        x, s, c, AEm = Generator(env)
        # Store the results in the dictionary.
        data[i][j] = {'x': x,
                      's': s,
                      'c': c,
                      'AEm': AEm,
                      'bimodality':bimodality_coefficient(x),
                      'avg_c':c.mean(),
                      'avg_AE':AEm.mean() if AEm is not None else np.nan
                      }

torch.save(data, 'results/data-final.pt')

