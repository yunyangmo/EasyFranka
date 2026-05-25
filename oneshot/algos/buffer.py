import numpy as np
import torch
import math
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def __len__(self):
        return len(self.buffer)
    def add(self, state, action, reward, next_state, done):
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state.detach(), action, reward, next_state.detach(), done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    def load(self, data):
        states = data["obs_seq"]    # point cloud (2048, 3)
        actions = data["actions_seq"]
        joints = data["joints_seq"] # might be used
        rewards = data["rewards_seq"]
        next_states = data["next_obs_seq"]
        dones = data["dones_seq"]

        
        N = states.shape[0]
        self.time_steps = N
        for i in range(N):
            state = torch.tensor(states[i], dtype=torch.float32)
            action = torch.tensor(actions[i], dtype=torch.float32)
            reward = torch.float(rewards[i])       # scalar
            next_state = torch.tensor(next_states[i], dtype=torch.float32)
            done = torch.int(dones[i])            # bool scalar

            self.add(state, action, reward, next_state, done)


class RLPDBuffer(ReplayBuffer):
    """Replay buffer for RLPD algorithm."""
    def __init__(self, capacity):
        super().__init__(capacity)
        self.off_position = 0
        self.offline_buffer = []
    # def update_last_reward(self, )
    def update_offline_rewards(self, new_reward):
        for i in range(len(self.offline_buffer)):
            if self.offline_buffer[i] is not None:
                state, action, old_reward, next_state, done = self.offline_buffer[i]
                
                # 根据 new_reward 类型确定替换值
                if callable(new_reward):
                    # 如果是函数，传递整个样本元组
                    updated_reward = new_reward((state, action, old_reward, next_state, done))
                else:
                    updated_reward = new_reward
                    
                # 更新缓冲区中的样本
                self.offline_buffer[i] = (state, action, updated_reward, next_state, done)



    def add_offline_data(self, state, action, reward, next_state, done):
        if len(self.offline_buffer) < 10000:
            self.offline_buffer.append(None)
        self.offline_buffer[self.off_position] = (state.detach(), action, reward, next_state.detach(), done)
        self.off_position = (self.off_position + 1)
    
    def online_sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def offline_sample(self, batch_size):
        batch = random.sample(self.offline_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    def update(self, rlpd_buffer):
        self.train_step += 1

        # sample
        obs, action, reward, next_obs, done = rlpd_buffer.mix_sample(self.batch_size)
        not_done = 1 - done
                
        # to tensor
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(self.device)
        
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
            'alpha': self.alpha.item()
        }



    def mix_sample(self, batch_size, mix_ratio=0.5):
        online_batch_size = int(batch_size * mix_ratio)
        offline_batch_size = batch_size - online_batch_size
        
        online_data = self.online_sample(online_batch_size)
        offline_data = self.offline_sample(offline_batch_size)
        
        mixed_data = (
            np.concatenate((online_data[0], offline_data[0]), axis=0),
            np.concatenate((online_data[1], offline_data[1]), axis=0),
            np.concatenate((online_data[2], offline_data[2]), axis=0),
            np.concatenate((online_data[3], offline_data[3]), axis=0),
            np.concatenate((online_data[4], offline_data[4]), axis=0)
        )
        
        return mixed_data
    
    def load_offline_data(self, file_path):



        data = np.load(file_path, allow_pickle=True)

        states = data["obs_seq"]    # point cloud (2048, 3)
        actions = data["actions_seq"]
        rewards = data["rewards_seq"]
        next_states = data["next_obs_seq"]
        dones = data["dones_seq"]

        
        N = states.shape[0]
        self.time_steps = N
        for i in range(N):
            state = torch.tensor(states[i], dtype=torch.float32)
            action = torch.tensor(actions[i], dtype=torch.float32)
            reward = float(rewards[i])       # scalar
            next_state = torch.tensor(next_states[i], dtype=torch.float32)
            done = int(dones[i])            # bool scalar

            self.add_offline_data(state, action, reward, next_state, done)
