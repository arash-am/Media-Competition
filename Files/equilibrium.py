import torch
import matplotlib.pyplot as plt  # <-- for plotting
import scipy.stats as stats # <-- for bimodality

from Files.enviroment import *

# Make sure that a device is defined for both the environment and the simulation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bimodality_coefficient(data):
    """
    Compute the bimodality coefficient from the skewness and (non-Fisher) kurtosis.
    """
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=False)
    bc = (skewness**2 + 1) / kurtosis
    return bc

def run_simulation(i, j, env):
    """
    Reset the environment, create parallel vectors of action IDs (one per env) using the
    provided scalar choices (i and j), run the simulation (which steps for terminal_time iterations)
    and return the average reward.
    """
    env.reset()
    action_ids_player = i * torch.ones(env.num_envs, device=device, dtype=torch.int64)
    action_ids_opponent = j * torch.ones(env.num_envs, device=device, dtype=torch.int64)

    # Run the simulation; the new simulate() method runs a full trajectory and returns rewards.
    rewards = env.simulate(action_ids_player, action_ids_opponent)
    return rewards.mean().item()

def QRE(A, taw=0.1, device=device):
    """
    Compute a Quantal Response Equilibrium (QRE) solution for a given payoff matrix A.
    (This is essentially unchanged except for minor clarifications and explicit normalization.)
    """
    Ar = A / A.abs().max()
    mu = torch.softmax(torch.rand((Ar.shape[0], 1), device=device), dim=0)
    nu = torch.softmax(torch.rand((Ar.shape[0], 1), device=device), dim=0)

    teta = taw / 30
    for _ in range(3 * 10**4):
        mu_old = mu.clone()
        nu_old = nu.clone()

        mub = mu.pow(1 - teta * taw) * torch.exp(teta * (Ar @ nu))
        mub = mub / mub.sum()
        nub = nu.pow(1 - teta * taw) * torch.exp(-teta * (Ar.t() @ mu))
        nub = nub / nub.sum()

        mu = mu.pow(1 - teta * taw) * torch.exp(teta * (Ar @ nub))
        mu = mu / mu.sum()
        nu = nu.pow(1 - teta * taw) * torch.exp(-teta * (Ar.t() @ mub))
        nu = nu / nu.sum()

        # Compute the duality gap; stop if it is sufficiently small.
        DG = (nu_old.t() @ Ar @ mu) - (nu.t() @ Ar @ mu_old)
        if DG.abs().item() < 1e-7:
            break
    return mu, nu

def Generator(env, Taw=0.1):
    """
    Load the payoff matrix A, compute a QRE (which gives a probability distribution over actions),
    instantiate the updated environment OpinionEnv9Actions, sample one action (per “player”) from the QRE,
    run a simulation, and return the final opinions x, susceptibilities s, and credibility c.
    (The fourth returned value is set to None because the previous code returned env.AEm which is no longer defined.)
    """
    # Load the payoff matrix from file.
    load_file=f'Payoff_matrix/Payoff-eta1_{round(env.eta, 1)}eta2_{round(env.eta2, 1)}.pt'
    A = torch.load(load_file)
    M = 10
    N = 500
    terminal_time = 200
    # Compute the QRE equilibrium strategies.
    mu, nu = QRE(A, taw=Taw, device=device)
    # Sample one action index for each of the two “players” from the QRE distributions.
    # We use num_samples=1 so that a single action is applied to all parallel environments.
    action_i = torch.multinomial((mu / mu.sum()).view(-1), num_samples=1, replacement=True).item()
    action_j = torch.multinomial((nu / nu.sum()).view(-1), num_samples=1, replacement=True).item()
    # Run the simulation using the selected actions.
    run_simulation(action_i, action_j, env)
    # Extract the final state components.
    # In the new environment, opinions are stored in env.x, credibility in env.c, and susceptibility in env.s.
    x = env.x.cpu().numpy().flatten()
    s = env.s.cpu().numpy().flatten()
    c = env.c.cpu().numpy().flatten()
    return x, s, c, None


def Generator_beta(env, Taw=0.1):
    """
    Load the payoff matrix A, compute a QRE (which gives a probability distribution over actions),
    instantiate the updated environment OpinionEnv9Actions, sample one action (per “player”) from the QRE,
    run a simulation, and return the final opinions x, susceptibilities s, and credibility c.
    (The fourth returned value is set to None because the previous code returned env.AEm which is no longer defined.)
    """
    # Load the payoff matrix from file.
    load_file=f'Payoff_matrix_susceptiblity/Payoff-beta_1_{round(env.beta_1, 1)}beta_2_{round(env.beta_2, 1)}.pt'
    A = torch.load(load_file)
    M = 10
    N = 500
    terminal_time = 200

    # Compute the QRE equilibrium strategies.
    mu, nu = QRE(A, taw=Taw, device=device)
    # Sample one action index for each of the two “players” from the QRE distributions.
    # We use num_samples=1 so that a single action is applied to all parallel environments.
    action_i = torch.multinomial((mu / mu.sum()).view(-1), num_samples=1, replacement=True).item()
    action_j = torch.multinomial((nu / nu.sum()).view(-1), num_samples=1, replacement=True).item()
    # Run the simulation using the selected actions.
    run_simulation(action_i, action_j, env)
    # Extract the final state components.
    # In the new environment, opinions are stored in env.x, credibility in env.c, and susceptibility in env.s.
    x = env.x.cpu().numpy().flatten()
    s = env.s.cpu().numpy().flatten()
    c = env.c.cpu().numpy().flatten()
    return x, s, c, None
