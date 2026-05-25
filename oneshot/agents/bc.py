import numpy as np
import torch
import math
import random
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
from copy import deepcopy

from oneshot.algos.pointnet2 import pointnet2
from oneshot.algos.net import GaussianActor,DeterministicActor

class Behavior_Clone(nn.Module):
    def __init__(self, obs_dim, action_dim, batch_size, device, lr=3e-4):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        
        # pointnet++
        self.pc_extractor = pointnet2(13).to(self.device)  # 13 is the number of input channels, including xyz and extra features
        checkpoint = torch.load("/home/yun4/workspace/Real_Franka/best_model.pth")
        self.pc_extractor.load_state_dict(checkpoint["model_state_dict"])
        self.pc_extractor.eval()
        self.post_pn2 =  torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU()
            ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.post_pn2.parameters()), 
            lr=lr
            )

        self.register_buffer("action_mean", torch.zeros(1, 7))
        self.register_buffer("action_std", torch.ones(1, 7))
        self.to(self.device)

    def infer_pointnet2(self, xyz):
        
        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        if len(xyz.shape) == 2:
            xyz = xyz.unsqueeze(0)
        assert xyz.shape[2] == 3, "input xyz should have shape (b,N,3)"
        xyz = xyz.transpose(2, 1)
        extra_feat = torch.zeros(xyz.shape[0], 6, 2048).to(self.device)  # (batch, 6, 2048)
        xyz_feat = torch.cat([xyz, extra_feat], dim=1).to(self.device) # shape = (batch, 9, 2048)

        # forwarding
        _, l4_points = self.pc_extractor(xyz_feat)
        pooled_feat = torch.mean(l4_points, dim=2).to(self.device)
        out_feat = self.post_pn2(pooled_feat.detach())

        out_feat = out_feat.reshape((xyz.shape[0],-1)).to(self.device)
        assert out_feat.shape[-1] == 1024
        return out_feat

    def infer_timestep(self, timestep):
        timestep = timestep.unsqueeze(-1)
        rel_time = timestep / 200  # [0,1]
        
        time_embed = torch.cat([
            torch.sin(2 * np.pi * rel_time),
            torch.cos(2 * np.pi * rel_time)
        ], dim=-1)

        return time_embed
    
    def act(self, obs, deterministic=False):
        
        pc = obs["pc"] if isinstance(obs, dict) else obs
        q = torch.tensor(obs["q"],dtype=torch.float32,device=self.device) if isinstance(obs, dict) else None
        t =  torch.tensor(obs["t"],dtype=torch.float32,device=self.device) if isinstance(obs, dict) else None
        t = self.infer_timestep(t)
        pc = torch.tensor(pc, dtype=torch.float32, device=self.device)
        if len(pc.shape) == 3:  # (1, 2048, 3)
            pc = pc.squeeze(0)  # (2048, 3)
        if len(t.shape) == 1:  
            t = t.unsqueeze(0)
        # pc = self.infer_pointnet2(pc)
        obs = torch.cat([q.unsqueeze(0),t], dim=1) if q is not None else pc
        self.actor.eval()
        with torch.no_grad():
            # mean, std = self.actor(obs)
            # dist = Normal(mean, std)
            # if deterministic:
            #     action = mean
            # else:
            #     action = dist.sample()
            action_pred = self.actor.act(obs, deterministic=False)
            action = action_pred * self.action_std + self.action_mean

        return action.cpu().numpy().squeeze()
    
    def _update_action_norm(self, action_batch):
        """batch norm"""
        if torch.norm(self.action_mean) < 1e-6: # overfitting agent
            self.action_mean.copy_(action_batch.mean(dim=0, keepdim=True))
            self.action_std.copy_(action_batch.std(dim=0, keepdim=True) + 1e-8)
    
    def train_step(self, obs_batch,joints_batch,ts_batch, action_batch):
        """
        obs_batch: shape [B, obs_dim]
        action_batch: shape [B, action_dim]
        """
        obs_batch = obs_batch.to(dtype=torch.float32, device=self.device)
        action_batch = action_batch.to(dtype=torch.float32, device=self.device)
        joints_batch = joints_batch.to(dtype=torch.float32, device=self.device)
        ts_batch = ts_batch.to(dtype=torch.float32, device=self.device)
        ts_batch = self.infer_timestep(ts_batch)
        # obs_batch = self.infer_pointnet2(obs_batch)
        if len(ts_batch.shape) == 1:
            ts_batch = ts_batch.unsqueeze(0)
        obs_batch = torch.cat([joints_batch,ts_batch], dim=1)

        # batch norm
        self._update_action_norm(action_batch)
        action_batch_norm = (action_batch - self.action_mean) / self.action_std

        ### frozen std?

        # mean, std = self.actor(obs_batch)
        # dist = Normal(mean, std)a
        # log_probs = dist.log_prob(action_batch).sum(axis=-1)  # sum over action_dim
        # loss = -log_probs.mean()  # maximize log likelihood == minimize negative log likelihood
        self.actor.train()
        sampled_actions = self.actor.rsample(obs_batch)
        
        log_probs,log_probs_dims = self.actor.log_prob(obs_batch, sampled_actions)
        
        expert_log_probs, _ = self.actor.log_prob(obs_batch, action_batch_norm)
        
        kl_div = (log_probs - expert_log_probs).mean()
    
        entropy = self.actor.entropy(obs_batch).mean()

        loss = kl_div - 0.01 * entropy

        action_pred_norm, std = self.actor(obs_batch)


        xyz_loss = ((action_pred_norm[:, 0:3] - action_batch_norm[:, 0:3]) ** 2).mean()
        euler_loss = ((action_pred_norm[:, 3:6] - action_batch_norm[:, 3:6]) ** 2).mean()
        gripper_loss = ((action_pred_norm[:, 6] - action_batch_norm[:, 6]) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss":loss.item(),
            "xyz_loss":xyz_loss.item(),
            "euler_loss":euler_loss.item(),
            "gripper_loss":gripper_loss.item(),
            "std": log_probs_dims.exp().clone()
            }