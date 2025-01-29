import numpy as np # used for arrays
import math        # needed for calculations
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Opinion_w_media():
    def __init__(self,
                 N=10**3,
                 M=10,
                 terminal_time=500,
                 bM=5,
                 b=25,
                 noise_level=0.1,
                 duration=50,
                 nbins=10,
                 h=0.01,
                 r_scale=100,
                 eta_1=1,
                 eta_2=1,
                 beta_1=3,
                 beta_2=2,):
        self.N = N
        self.M = M
        self.terminal_time = terminal_time
        Beta = torch.distributions.beta.Beta(beta_1,beta_2)
        # Put parameters on the same device:
        self.s = Beta.sample(sample_shape=(N,)).to(device)
        self.ym = torch.linspace(-1, 1, steps=self.M).to(device)
        self.noise_level = noise_level
        self.bM = bM
        self.b = b
        self.duration = duration
        self.action_dim = 9   # e.g. 2^(M/2) if M is even
        self.nbins = nbins
        self.h = h
        self.pi = torch.tensor(math.pi, device=device)
        self.r_scale = r_scale
        self.eta_1=eta_1
        self.eta_2 = eta_2
        self.AEm_abs = torch.zeros((self.N, self.M), device=device)
    def reset(self):
        """
          Resets the environment.
          state = [ x (N), c (M), t (scalar) ]
        """
        x_init = (torch.rand(self.N, device=device) * 2 - 1)
        c_init = torch.ones(self.M, device=device)
        t_init = torch.tensor([0.0], device=device)
        self.state = torch.cat([x_init, c_init, t_init])
        return self.state

    def action_list(self, action_ids):
        zm = torch.linspace(0, 1, steps=int(self.M / 2)).to(self.device)
        # action_ids = action_ids.unsqueeze(1)
        probabilities = torch.tensor([.95, .95, .95, .5, .5, .1, .1, .5, .1], device=self.device)
        deltas = torch.tensor([.000001, .85, .45, -.00001, -.45, -.85, -.0000001, .4, -.4], device=self.device)
        base_probs = probabilities[action_ids].squeeze()
        delta_probs = deltas[action_ids].squeeze()
        return torch.bernoulli(base_probs - zm[:, None] * delta_probs)

    def dyn_step(self, u_action_id, v_action_id):
        """
        The main dynamic step of the system for self.duration steps.
        """
        done = False
        state = self.state
        reward = 0.0
        h = self.h
        gam = 0.98

        u_actions = self.action_list(u_action_id).t()
        v_actions = self.action_list(v_action_id).t()
        action = torch.cat([u_actions, v_actions], dim=1)
        # Run multiple sub-steps in the environment
        for _ in range(self.duration):
            x, c, t = state[:self.N], state[self.N:-1], state[-1]
            DM = self.ym - x[:, None]    # shape [N, M]
            D  = x - x[:, None]         # shape [N, N]

            # shape [N, M], using broadcast
            AM = torch.exp(-self.bM * (1 + self.eta_1*action) * (1 + self.eta_2*(2 - c - action)*(1 - self.s.reshape(-1,1)))
                           * torch.abs(DM))
            # shape [N, N]
            A = torch.exp(-self.b * torch.abs(D))
            self.AEm_abs += torch.exp(-self.bM * torch.abs(DM)) * (1 - action)
            t += 1
            # Weighted average for credibility
            c = c * (gam) + action * (1 - gam)

            # Update x
            x += h * (
                (torch.sum(A * D, dim=1) / torch.sum(A, dim=1))
              + (torch.sum(AM * DM, dim=1) / torch.sum(AM, dim=1))
            ) + torch.sqrt(h)*torch.normal(0, self.noise_level, size=(self.N,), device=device)

            state = torch.cat([x, c, t.unsqueeze(0)])

        self.state = state
        # Negative of a certain function of x
        reward = self.r_scale * (-1 * (torch.sin(self.pi*x/2)).pow(5)).mean()

        if state[-1] >= self.terminal_time:
            done = True
        return reward, done

    def state2obs(self, state):
        """
        Observations = histogram of x (size nbins) + c (size M).
        """
        xs = state[:self.N].clone().detach()
        cs = state[self.N:self.N+self.M].clone().detach()
        # Make sure histogram is on GPU; use torch.histc
        hist_x = torch.histc(xs, bins=self.nbins, min=-1, max=1)
        hist_x = hist_x / self.N
        return torch.cat([hist_x, cs])

    def step(self, u_action_id, v_action_id):
        """
        Perform a single environment step:
         returns next_state, reward, done, {}
        """
        reward, done = self.dyn_step(u_action_id, v_action_id)
        return self.state, reward, done, None

    def plot_dist(self, x, c):
        """
        Visualization helper
        """
        cmap = plt.cm.coolwarm
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(self.ym.cpu(), c.cpu(), 50, c=self.ym.cpu(), cmap=cmap)
        plt.title('Credit Bias')
        plt.xlabel("Opinion-Bias")
        plt.ylabel("Credit Score")
        plt.xlim([-1.1, 1.1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.hist(x.cpu(), bins=np.linspace(-1.2, 1.2, 50), color='skyblue',
                 edgecolor='black', alpha=0.3, density=True)
        plt.scatter(x.cpu(), self.s.cpu(), 50, c=x.cpu(), cmap=cmap, alpha=.1)
        plt.ylabel("Opinion")
        plt.xlabel("Frequency")
        plt.title("Distribution of Final Opinions")
        plt.grid(True)
        plt.xlim([-1.3, 1.3])
        plt.tight_layout()
        plt.show()
