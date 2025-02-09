import os
import numpy as np
import torch
from tqdm import tqdm

from Files.equilibrium import *


# Parameters
M = 10
N = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta_list = np.arange(0, 4.1, 0.1)


# Loop over eta1 and eta2 values with progress bars
for beta_1 in tqdm(beta_list, desc='Outer Loop Progress'):
    for beta_2 in tqdm(beta_list, desc=f'Inner Loop (eta1={beta_1:.1f})', leave=False):
        # Instantiate the environment with the current eta1 and eta2 values
        env = OpinionEnv9Actions(
            num_envs=500,
            N=N,
            M=M,
            terminal_time=200,
            bM=5,
            b=20,
            noise_level=0.1,
            h=0.1,
            nbins=30,
            r_scale=100,
            eta=1,
            eta2=1,
            beta_1=beta_1,
            beta_2=beta_2,
            lambda_c=0.98
        )
        # Initialize the payoff matrix A
        A = torch.zeros((env.action_dim, env.action_dim), device=device)
        # Populate A by running simulations for each pair of actions
        for i in range(env.action_dim):
            for j in range(env.action_dim):
                A[i, j] = run_simulation(i, j, env)
        # Save the payoff matrix to a file with a descriptive name
        save_file = f'Payoff_matrix_susceptiblity/Payoff-beta_1_{round(beta_1, 1)}beta_2_{round(beta_2, 1)}.pt'
        torch.save(A, save_file)



os.makedirs('results_susceptibility', exist_ok=True)
# Define the ranges for eta values.
data = {}       # Dictionary to store full simulation output
# Loop over the grid of eta values with nested progress bars.

for i, beta_1 in tqdm(enumerate(beta_list), total=len(beta_list), desc="Outer Loop (Eta1)"):
    data[i] = {}
    for j, beta_2 in tqdm(enumerate(beta_list), total=len(beta_list), desc=f"Inner Loop (eta1={beta_1:.1f})", leave=False):
        # Run the simulation for the current eta1 and eta2 combination.
        env.beta_2= beta_1
        env.beta_1 = beta_2
        env.reset()
        x, s, c, AEm = Generator_beta(env)
        # Store the results in the dictionary.
        data[i][j] = {'x': x,
                      's': s,
                      'c': c,
                      'AEm': AEm,
                      'bimodality':bimodality_coefficient(x),
                      'avg_c':c.mean(),
                      'avg_AE':AEm.mean() if AEm is not None else np.nan
                      }

torch.save(data, 'results_susceptibility/data-final.pt')

