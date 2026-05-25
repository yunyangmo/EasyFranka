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
import torch.optim as optim
import numpy as np
from copy import deepcopy

from oneshot.algos.net import SACActor, Critic, Critic_layernorm
from oneshot.algos.pointnet2 import pointnet2

class SACAgent:
    """SAC algorithm with joint limit handling."""
    def __init__(self, obs_dim, action_dim, action_range, device,
                 discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        # params
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # action range
        # delta xyz, euler, gripper_pos
        self.low = torch.tensor([-0.1*3, -0.785*3, 0.], device=device, dtype=torch.float32)
        self.high = torch.tensor([0.1*3, 0.785*3, 200.], device=device, dtype=torch.float32)
        

        
        

        hidden_dim = 256 
        self.actor = SACActor(obs_dim, action_dim, hidden_dim, self.low, self.high).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        self.pointnet2_init()

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(actor_lr), betas=actor_betas
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=float(critic_lr), betas=critic_betas
        )
        
        # temperature learnable
        self.log_alpha = torch.tensor(np.log(init_temperature), device=device)
        self.log_alpha.requires_grad = learnable_temperature
        if learnable_temperature:
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=float(alpha_lr), betas=alpha_betas
            )
        
  
        self.train_step = 0
        self.update_step = 0
        
        self.target_entropy = -action_dim  
        
    @property
    def alpha(self):
        return self.log_alpha.exp().detach()
    def pointnet2_init(self):
        self.pc_extractor = pointnet2(13)
        checkpoint = torch.load("/home/yun4/workspace/Real_Franka/best_model.pth")
        self.pc_extractor.load_state_dict(checkpoint["model_state_dict"])
        self.pc_extractor.eval()

        self.post_pn2 =  torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU()
            )
    def infer_pointnet2(self, xyz):
        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)

        assert xyz.shape[1] == 3, "input xyz should have shape (N,3)"
        xyz = xyz.transpose(0, 1).unsqueeze(0)
        extra_feat = torch.zeros(1, 6, 2048)
        xyz_feat = torch.cat([xyz, extra_feat], dim=1)  # shape = (1, 9, 2048)
        _, l4_points = self.pc_extractor(xyz_feat)
        pooled_feat = torch.mean(l4_points, dim=2)
        out_feat = self.post_pn2(pooled_feat.detach())

        return out_feat

    def select_action(self, obs, sample=False):
        
        obs = self.infer_pointnet2(obs) #(b, 1024)

        if sample:
            with torch.no_grad():
                action, _ = self.actor(obs)
            action = action.squeeze(0).cpu().numpy()
        else:

            with torch.no_grad():
                action = self.actor(obs, compute_logprob=False)
            action = action.squeeze(0).cpu().numpy()
        

        return np.clip(action, self.low.cpu().numpy(), self.high.cpu().numpy())
    
    def update_critic(self, obs, action, reward, next_obs, not_done):


        if len(obs.shape) != 2:
            obs = obs.unsqueeze(0)

        with torch.no_grad():

            next_action, next_log_prob = self.actor(next_obs)
            

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob.unsqueeze(1)
            target_value = reward + not_done * self.discount * target_Q
        

        current_Q1, current_Q2 = self.critic(obs, action)
        

        critic_loss = F.mse_loss(current_Q1, target_value) + F.mse_loss(current_Q2, target_value)
        

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor_and_alpha(self, obs):

        for param in self.critic.parameters():
            param.requires_grad = False
        

        new_action, log_prob, _ = self.actor(obs)
        

        Q1, Q2 = self.critic(obs, new_action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_prob.unsqueeze(1) - Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True
        
        alpha_loss = 0.0
        if self.learnable_temperature:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        return actor_loss.item(), alpha_loss.item()
    
    def soft_update_critic_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data
            )
    
    def update(self, replay_buffer):

        self.train_step += 1

        # sample
        obs, action, reward, next_obs, done = replay_buffer.sample(self.batch_size)
        not_done = 1 - done
                
        # (b, dims)
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        not_done = torch.FloatTensor(not_done).unsqueeze(-1).to(self.device)
        
        if len(obs.shape) != 2:
            obs = obs.view(obs.shape[0], -1)
        if len(next_obs.shape) != 2:
            next_obs = next_obs.view(next_obs.shape[0], -1)


        # update critic
        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done)
        

        actor_loss, alpha_loss = 0.0, 0.0
        if self.train_step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs)
        

        if self.train_step % self.critic_target_update_frequency == 0:
            self.soft_update_critic_target()
 
        if actor_loss > 0 or alpha_loss > 0:
            self.update_step += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
        }
    
    
    
    def save(self, filename):

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.learnable_temperature else None,
            'train_step': self.train_step,
            'update_step': self.update_step
        }, filename)
    
    def load(self, filename):

        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.learnable_temperature and checkpoint['alpha_optimizer'] is not None:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.train_step = checkpoint['train_step']
        self.update_step = checkpoint['update_step']
