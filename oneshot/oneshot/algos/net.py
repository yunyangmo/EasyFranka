import numpy as np
import torch
import math
import random
from torch import nn 
import torch.nn.functional as F
from torch import distributions as pyd
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        action = self.net(obs)
        return action
    
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self._initialize_weights()
        # nn.init.zeros_(self.mean_layer.weight)
        # nn.init.zeros_(self.mean_layer.bias)
        # nn.init.zeros_(self.log_std_layer.weight)
        # nn.init.zeros_(self.log_std_layer.bias)
    
    def _initialize_weights(self):
       
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.1)
        
        nn.init.constant_(self.log_std_layer.weight, 0.001)
        nn.init.constant_(self.log_std_layer.bias, -1.0)   
    
    def forward(self, obs, return_distribution=False):

        features = self.net(obs)
        mean = self.mean_layer(features)

        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, -2)  # 避免极端标准差
        std = log_std.exp()
        
        if return_distribution:
            return pyd.Normal(mean, std)
        return mean, std
    def rsample(self, obs):
        dist = self.forward(obs, return_distribution=True)
        return dist.rsample()
    def log_prob(self, obs, action):
        dist = self.forward(obs, return_distribution=True)
        return dist.log_prob(action).sum(dim=-1),dist.log_prob(action)
    def entropy(self, obs):
        dist = self.forward(obs, return_distribution=True)
        return dist.entropy().sum(dim=-1)
    def act(self, obs, deterministic=False):

        if deterministic:
            mean, _ = self.forward(obs)
            return mean
        else:
            dist = self.forward(obs, return_distribution=True)
            return dist.sample()
    

class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, low, high):
        super().__init__()
        assert len(low) == action_dim and len(high) == action_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.register_buffer("low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(high, dtype=torch.float32))

        
        self.register_buffer("scale", (self.high - self.low) / 2.0)
        self.register_buffer("bias", (self.high + self.low) / 2.0)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_action(self, obs):
        x = self.trunk(obs)
        mean = self.mean(x)
        action_tanh = torch.tanh(mean)  # range [-1, 1]
        scaled_action = action_tanh * self.scale + self.bias  # 2 [low, high]
        return scaled_action

    def forward(self, obs, compute_logprob=True):
        x = self.trunk(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()

        action_tanh = torch.tanh(z)
        scaled_action = action_tanh * self.scale + self.bias

        if compute_logprob:
            log_prob = normal.log_prob(z).sum(dim=-1)

            # tanh calibrate log-prob
            jacobian = 2 * (np.log(2) - z - F.softplus(-2 * z))
            log_prob -= jacobian.sum(dim=-1)

            return scaled_action, log_prob, normal.entropy().mean()
        else:
            return scaled_action


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def random_init(m):
        if isinstance(m, nn.Linear):
            # Kaiming
            nn.init.kaiming_uniform_(m.weight.data, 
                                    a=0,          
                                    mode='fan_in', 
                                    nonlinearity='relu')

            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)

class Critic_layernorm(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)

class Single_Critic_ln(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.Q = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self.weight_init)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.Q(x)