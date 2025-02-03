import math
import numpy as np
import torch

# ----------------------
# Check Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------
# 1) Environment with 9 Discrete Actions
# ----------------------
class OpinionEnv9Actions:
    """
    A vectorized environment with exactly 9 discrete actions for each "player" (row & column).
    Each action maps to M/2 Bernoulli bits using 'probabilities' and 'deltas', then we concatenate
    row & column to form an M-dimensional 'action' for the media update.

    The adjacency factor for media is:
        fac = (1 + eta * action) * (1 + eta2 * (2 - c - action)*(1 - s))

    Credibility is updated with: c = gamma * c + (1 - gamma)*action
    The observation is a histogram of x plus the current credibility c.
    """

    def __init__(self,
                 num_envs=1000,
                 N=500,
                 M=10,
                 terminal_time=200,
                 bM=4,
                 b=12,
                 noise_level=0.12,
                 h=0.1,
                 nbins=30,
                 r_scale=100,
                 eta=1,
                 eta2=2,
                 beta_1=3,
                 beta_2=2,
                 lambda_c=0.98):
        """
        :param num_envs:       Number of parallel envs
        :param N:              Number of agents
        :param M:              Media dimension
        :param terminal_time:  Steps until done
        :param bM, b:          Coupling constants for media and social
        :param noise_level:    Std dev of noise
        :param h:              Euler step size
        :param nbins:          Number of histogram bins
        :param r_scale:        Scale factor for rewards
        :param eta, eta2:      Multipliers in the adjacency factor
        :param beta_1, beta_2: Beta distribution parameters for susceptibility
        :param gamma:          Credibility discount factor
        """
        self.num_envs = num_envs
        self.N = N
        self.M = M
        self.terminal_time = terminal_time
        self.bM = bM
        self.b = b
        self.noise_level = noise_level
        self.h = torch.tensor(h, device=device, dtype=torch.float32)
        self.nbins = nbins
        self.r_scale = r_scale
        self.eta = eta
        self.eta2 = eta2
        self.lambda_c = lambda_c
        self.pi = torch.tensor(math.pi, device=device)

        # Beta distribution for s
        BetaDist = torch.distributions.beta.Beta(beta_1, beta_2)

        # Shared across all envs: media positions in [-1,1], shape [M]
        self.ym = torch.linspace(-1, 1, steps=self.M, device=device)

        # Environment states: x, c, t, s
        # x: shape [num_envs, N]
        # c: shape [num_envs, M]
        # t: shape [num_envs]
        # s: shape [num_envs, N] (from BetaDist)
        self.x = torch.zeros(self.num_envs, self.N, device=device)
        self.c = torch.zeros(self.num_envs, self.M, device=device)
        self.t = torch.zeros(self.num_envs, device=device)
        self.s = BetaDist.sample(sample_shape=(self.num_envs, self.N)).to(device)

        # Exactly 9 actions
        self.action_dim = 9

        # Prob & delta for each of the 9 discrete actions
        self.probabilities = torch.tensor(
            [0.95, 0.95, 0.95, 0.50, 0.50, 0.10, 0.10, 0.50, 0.10],
            device=device
        )
        self.deltas = torch.tensor(
            [0.000001, 0.85, 0.45, -0.00001, -0.45, -0.85, -0.0000001, 0.4, -0.4],
            device=device
        )
        # For the M/2 dimension
        self.zm = torch.linspace(0, 1, steps=self.M // 2, device=device)

        # Init
        self.reset()

    def reset(self):
        """
        Reset all envs to initial state.
        """
        with torch.no_grad():
            # x in [-1,1]
            self.x.uniform_(-1.0, 1.0)
            # c=1
            self.c.fill_(1.0)
            # t=0
            self.t.zero_()
            # If you want to re-sample s here, uncomment:
            BetaDist = torch.distributions.beta.Beta(3,2)
            self.s = BetaDist.sample(sample_shape=(self.num_envs, self.N)).to(device)
        return self.state2obs()

    def state2obs(self):
        """
        Observation shape [num_envs, nbins + M] = [hist(x), c].
        """
        obs_list = []
        for i in range(self.num_envs):
            # histogram of x[i] with nbins in [-1,1], normalized
            hist_i = torch.histc(self.x[i], bins=self.nbins, min=-1, max=1)
            hist_i = hist_i / self.N
            # c[i]: shape [M]
            obs_list.append(torch.cat([hist_i, self.c[i]]))
        # stack => shape [num_envs, nbins+M]
        obs = torch.stack(obs_list, dim=0)
        return obs

    def _convert_action_ids_to_vec(self, action_ids):
        """
        Convert each action_id in {0..8} to an M/2 Bernoulli vector.
        final_prob = base_prob - zm * delta_prob
        Then sample Bernoulli( final_prob ).
        Returns shape [num_envs, M//2].
        """
        base_probs = self.probabilities[action_ids]  # [num_envs]
        delta_probs = self.deltas[action_ids]        # [num_envs]

        # final_prob[i,j] = base_probs[i] - (zm[j]*delta_probs[i])
        final_prob = base_probs.unsqueeze(1) - self.zm.unsqueeze(0)*delta_probs.unsqueeze(1)
        final_prob = final_prob.clamp(min=0.0, max=1.0)
        action_vec = torch.bernoulli(final_prob)
        return action_vec  # [num_envs, M//2]

    def simulate(self, action_pl_ids, action_op_ids):
        """
        Step the environment by 1 Euler iteration using row/column action IDs in {0..8}.
        Returns next_obs, rewards, done, info.
        """
        with torch.no_grad():
            for _ in range(self.terminal_time):
                # Build row & column actions => [num_envs, M//2]
                row_act = self._convert_action_ids_to_vec(action_pl_ids)
                col_act = self._convert_action_ids_to_vec(action_op_ids)
                # Combine => [num_envs, M]
                action = torch.cat([row_act, col_act], dim=1)
                # Distances
                DM = self.ym.view(1,1,-1) - self.x.unsqueeze(2)  # [num_envs,N,M]
                xx = self.x.unsqueeze(2)
                D = xx - xx.transpose(1,2)                      # [num_envs,N,N]

                # Media adjacency factor:
                # fac = (1 + eta*action) * (1 + eta2*(2-c-action)*(1-s))
                # We'll do shape expansions:
                #   action => [num_envs,M] => unsqueeze(1)->[num_envs,1,M]
                  # (2-c-action) => same shape => unsqueeze(1)->[num_envs,1,M]
                #   (1-s) => [num_envs,N] => unsqueeze(2)->[num_envs,N,1]
                fac = (1.0 + self.eta*action).unsqueeze(1) * (
                    1.0 + self.eta2*(2.0 - self.c - action).unsqueeze(1)*(1.0 - self.s).unsqueeze(2)
                )

                AM = torch.exp(-self.bM * fac * torch.abs(DM))  # [num_envs,N,M]
                A = torch.exp(-self.b * torch.abs(D))           # [num_envs,N,N]

                # Credibility update => c = gamma*c + (1-gamma)*action
                self.c = self.c*self.lambda_c + action*(1-self.lambda_c)

                # Weighted updates for x D has sign in it
                x_update_social = -(A * D).sum(dim=2)/A.sum(dim=2).clamp_min(1e-12)
                x_update_media = (AM * DM).sum(dim=2)/AM.sum(dim=2).clamp_min(1e-12)

                # Noise
                noise = torch.normal(
                    mean=0.0, std=self.noise_level,
                    size=(self.num_envs, self.N),
                    device=device
                )

                # Euler update
                self.x = self.x + self.h*(x_update_social + x_update_media) + torch.sqrt(self.h)*noise

                # time
                self.t += 1

                # Reward => r_scale * -avg( sin^5(pi*x/2) )
                sin_val = torch.sin((self.pi/2.0)*self.x)
                sin_pow5 = sin_val.pow(5)
                mean_sin_pow5 = sin_pow5.mean(dim=1)  # [num_envs]
                rewards = self.r_scale * (-mean_sin_pow5)

                # Done?

        return rewards