import os
import numpy as np
import torch
from tqdm import tqdm

from Files.equilibrium import *



# Parameters
M = 10
N = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
taw_list = np.arange(-1, 2.1, 0.1)

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
    beta_1=3,
    beta_2=2,
    lambda_c=0.98
)


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

