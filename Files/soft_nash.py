import numpy as np # used for arrays
import math        # needed for calculations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        # Store as numpy for convenience; we move to device in the training step
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (np.concatenate(observations, 0),
                actions,
                rewards,
                np.concatenate(next_observations, 0),
                dones)

    def __len__(self):
        return len(self.memory)

class soft_q_net(nn.Module):
    def __init__(self, observation_dim, bpl, bop, M, action_dim=32):
        super(soft_q_net, self).__init__()
        self.observation_dim = observation_dim
        self.bpl = bpl
        self.bop = bop
        self.M = M
        self.action_dim = action_dim

        # Precompute indices for the row/col slices:
        self.makspl = torch.zeros(action_dim, action_dim, dtype=torch.long)
        self.maksop = torch.zeros(action_dim, action_dim, dtype=torch.long)
        idxs = torch.arange(action_dim, dtype=torch.long)
        for i in range(action_dim):
            self.makspl[i, :] = i * action_dim + idxs  # row combinations
            self.maksop[i, :] = action_dim * idxs + i  # column combinations

        # Move these to device, so indexing uses GPU
        self.makspl = self.makspl.to(device)
        self.maksop = self.maksop.to(device)

        # Network layers
        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, self.action_dim ** 2)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def log_sum_exp(x):
        """
        Numerically stable log-sum-exp along the last dimension.
        x shape: [batch, something]
        """
        max_x, _ = x.max(dim=1, keepdim=True)
        exps = torch.exp(x - max_x)
        sum_of_exps = exps.sum(dim=-1)
        return torch.log(sum_of_exps) + max_x.squeeze(dim=1)

    def getV(self, q_value):
        """
        V(s) = (1/bpl)* logsumexp( bpl * Q_pl(a) ) - log(action_dim).
        Q_pl has shape = [batch, action_dim].
        """
        Q_pl = self.getQ_pl(q_value)  # [batch, action_dim]
        # log-sum-exp over actions
        v = (1 / self.bpl) * (self.log_sum_exp(self.bpl * Q_pl) - math.log(self.action_dim))
        return v

    def getQ_pl(self, q_value):
        """
        For each row-action i, consider all column actions.
        q_value shape: [batch, action_dim^2].
        We slice out row i by the index self.makspl[i,:].
        Then do a log-sum-exp with factor bop for column dimension
        """
        batch_size = q_value.shape[0]
        Qa = torch.zeros((batch_size, self.action_dim), device=device)
        for i in range(self.action_dim):
            x = self.bop * q_value[ :, self.makspl[i, :] ]  # shape [batch, action_dim]
            # log-sum-exp over columns
            # (1/bop)*[ logsumexp(bop * Q(s,(row=i, col))) - log(action_dim) ]
            # minus log(action_dim) to keep it from blowing up or to keep it normalized
            # but code is consistent with the final expression
            Qa[:, i] = (1 / self.bop) * (
                self.log_sum_exp(x) - math.log(self.action_dim)
            )
        return Qa

    def getQ_op(self, q_value):
        """
        For each column-action i, consider all row actions.
        q_value shape: [batch, action_dim^2].
        We slice out column i by the index self.maksop[i,:].
        Then do a log-sum-exp with factor bpl for row dimension
        """
        batch_size = q_value.shape[0]
        Qa = torch.zeros((batch_size, self.action_dim), device=device)
        for i in range(self.action_dim):
            x = self.bpl * q_value[:, self.maksop[i, :]]  # shape [batch, action_dim]
            Qa[:, i] = (1 / self.bpl) * (
                self.log_sum_exp(x) - math.log(self.action_dim)
            )
        return Qa

    # def act(self, observation):
    #     """
    #     Sample an action for each player from the softmax distribution
    #     """
    #     with torch.no_grad():
    #         # Forward pass
    #         q_value = self.forward(observation)  # shape [batch, action_dim^2]
    #         Q_pl = self.getQ_pl(q_value)         # shape [batch, action_dim]
    #         Q_op = self.getQ_op(q_value)         # shape [batch, action_dim]
    #
    #         pi_pl = F.softmax(self.bpl * Q_pl, dim=-1)  # row player's policy
    #         pi_op = F.softmax(self.bop * Q_op, dim=-1)  # column player's policy
    #
    #         dist_pl = torch.distributions.Categorical(pi_pl)
    #         dist_op = torch.distributions.Categorical(pi_op)
    #
    #         # Sample a single action from each player
    #         ac_pl = dist_pl.sample()  # integer in [0, action_dim)
    #         ac_op = dist_op.sample()  # integer in [0, action_dim)
    #
    #         # Combine them into a single integer for indexing in Q(s,a)
    #         ac_id = ac_pl * self.action_dim + ac_op
    #
    #     # Return the integer action for each player, plus the combined id
    #     # and the distribution if you want it
    #     return ac_pl.item(), ac_op.item(), ac_id.item(), torch.cat([pi_pl[0], pi_op[0]])
    def act(self, observation, epsilon=0.1):
        """
        Sample an action for each player from the softmax distribution,
        but apply epsilon-greedy exploration with probability epsilon.
        """
        with torch.no_grad():
            # Forward pass
            q_value = self.forward(observation)  # shape [batch, action_dim^2]
            Q_pl = self.getQ_pl(q_value)  # shape [batch, action_dim]
            Q_op = self.getQ_op(q_value)  # shape [batch, action_dim]

            pi_pl = F.softmax(self.bpl * Q_pl, dim=-1)  # row player's policy
            pi_op = F.softmax(self.bop * Q_op, dim=-1)  # column player's policy

            dist_pl = torch.distributions.Categorical(pi_pl)
            dist_op = torch.distributions.Categorical(pi_op)

            # Apply epsilon-greedy exploration
            if random.random() < epsilon:
                ac_pl = torch.randint(0, self.action_dim, (1,), device=device).item()
                ac_op = torch.randint(0, self.action_dim, (1,), device=device).item()
            else:
                ac_pl = dist_pl.sample().item()
                ac_op = dist_op.sample().item()

            ac_id = ac_pl * self.action_dim + ac_op

        return ac_pl, ac_op, ac_id, torch.cat([pi_pl[0], pi_op[0]])

def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, count, update_freq, TAU):
    """
    One gradient update step from the replay buffer.
    """
    # Sample from replay
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    # Convert to torch and move to GPU
    observation      = torch.FloatTensor(observation).to(device)
    action           = torch.LongTensor(action).to(device)
    reward           = torch.FloatTensor(reward).to(device)
    next_observation = torch.FloatTensor(next_observation).to(device)
    done             = torch.FloatTensor(done).to(device)

    # Q-values for current state
    q_values = eval_model(observation)                # [batch, action_dim^2]
    # Q-values for next state (target net)
    next_q_values = target_model(next_observation)    # [batch, action_dim^2]
    next_v_values = target_model.getV(next_q_values)  # [batch]

    # Gather Q(s,a) based on the combined action index
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # Soft Q-learning target
    expected_q_value = reward + gamma * (1 - done) * next_v_values

    # Standard MSE loss
    loss = (expected_q_value.detach() - q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping if desired
    torch.nn.utils.clip_grad_norm_(eval_model.parameters(), max_norm=1.0)
    optimizer.step()

    # Soft update for target network
    with torch.no_grad():
        target_net_state_dict = target_model.state_dict()
        policy_net_state_dict = eval_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_model.load_state_dict(target_net_state_dict)

    # Some debug info
    with torch.no_grad():
        loss_p = loss
        # Normalized loss by Q variance
        q_var = q_value.var() if q_value.var() > 1e-8 else torch.tensor(1.0, device=device)
        loss_n = loss_p / q_var
    return loss_p.item(), loss_n.item(), next_q_values
