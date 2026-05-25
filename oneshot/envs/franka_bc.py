import os,sys,time,copy
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import datetime
import random
import threading
from collections import deque
from typing import Tuple
import requests
import yaml
from termcolor import cprint
import numpy as np
import cv2
import pyspacemouse
import pyrealsense2 as rs
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from oneshot.utils import Franka,RSCapture,SpaceMouseExpert
from oneshot.utils.operation import transform_camera_to_marker,interpolate_se3_euler
from oneshot.utils.camera import voxel_downsampling,furthest_point_sampling,PointNetEncoderXYZ,fast_furthest_point_sampling

from oneshot.agents import RLPDAgent, Q_RLPDAgent, Behavior_Clone
from oneshot.algos import ReplayBuffer, RLPDBuffer
import time

from oneshot.envs import FrankaEnv

def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    env = FrankaEnv()  
    # env.reset()
    writer = SummaryWriter(log_dir='runs/pnp/bc/01')
    

    with open("/home/yun4/workspace/Real_Franka/Reinforcement_Learning/oneshot/config/agent.yaml", "r") as f:
        config = yaml.safe_load(f)
    agent_config = config['agent']['params']


    ckpt_path = "/home/yun4/workspace/Real_Franka/Reinforcement_Learning/oneshot/dataset/bc/ckpt"
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_file = os.path.join(ckpt_path, "bc_agent.pth")
    

    state_path = "/home/yun4/workspace/Real_Franka/Reinforcement_Learning/oneshot/dataset/bc/state"
    
    # load data and copycat
    config['env']['offline_data_path']
    data = np.load(config['env']['offline_data_path'], allow_pickle=True)

    states = data["obs_seq"]    # point cloud (2048, 3)
    actions = data["actions_seq"]
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    joints = torch.tensor(data["joints_seq"], dtype=torch.float32)
    ts = torch.tensor(range(states.shape[0]), dtype=torch.float32)
    # torch loader
    dataset = TensorDataset(states, actions,joints, ts)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # bc agent
    agent = Behavior_Clone(obs_dim=7+2, action_dim=7, batch_size=256, device="cuda")


    # agent.load_state_dict(torch.load(os.path.join(ckpt_path, "q_t.pth"), map_location=agent.device))
    

    if os.path.exists(ckpt_file):
        print(f"权重文件已存在: {ckpt_file}，加载中...")
        agent.load_state_dict(torch.load(ckpt_file, map_location=agent.device))
    else:
        for epoch in range(800):
            epoch_loss = 0.0
            xyz_loss = 0.0
            euler_loss = 0.0
            gripper_loss = 0.0
            train_std = 0.0
            for obs_batch, action_batch,joints_batch,ts_batch in dataloader:
                loss_dict = agent.train_step(obs_batch,joints_batch,ts_batch, action_batch)
                
                epoch_loss += loss_dict["loss"]
                xyz_loss += loss_dict["xyz_loss"]
                euler_loss += loss_dict["euler_loss"]
                gripper_loss += loss_dict["gripper_loss"]
                train_std += loss_dict["std"]
            avg_loss = epoch_loss / len(dataloader)

            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
            writer.add_scalar("Loss/BC_train", avg_loss, epoch)
            writer.add_scalar("Loss/xyz_loss", xyz_loss/len(dataloader), epoch)
            writer.add_scalar("Loss/euler_loss", euler_loss/len(dataloader), epoch)
            writer.add_scalar("Loss/gripper_loss", gripper_loss/len(dataloader), epoch)
            mean_xyz_std = train_std[:, :3].mean()
            mean_euler_std = train_std[:, 3:6].mean()
            mean_gripper_std = train_std[:, -1].mean()
            writer.add_scalar("Loss/std", train_std.mean()/len(dataloader), epoch)
            writer.add_scalar("Loss/xyz_std", mean_xyz_std.mean()/len(dataloader), epoch)
            writer.add_scalar("Loss/euler_std", mean_euler_std.mean()/len(dataloader), epoch)
            writer.add_scalar("Loss/gripper_std", mean_gripper_std.mean()/len(dataloader), epoch)


        torch.save(agent.state_dict(), ckpt_file)
        print(f"✅ BC 模型权重已保存到 {ckpt_file}")

    # start_time = time.time()
    
    # for epoch in range(2000):
    #     epoch_start = time.time()
    #     epoch_loss = 0.0
        
    #     for obs_batch, action_batch,joints_batch,ts_batch in dataloader:
    #         loss = agent.train_step(obs_batch,joints_batch,ts_batch, action_batch)
    #         epoch_loss += loss

    #     avg_loss = epoch_loss / len(dataloader)
        
    #     writer.add_scalar("Loss/BC_train", avg_loss, 200+epoch)
        
    #     # ETA
    #     epoch_end = time.time()
    #     epoch_duration = epoch_end - epoch_start
    #     elapsed = epoch_end - start_time
    #     epochs_left = 2000 - (epoch + 1)
    #     est_remaining = epochs_left * epoch_duration
    #     print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
    #     print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | "
    #       f"Time: {epoch_duration:.2f}s | "
    #       f"Elapsed: {elapsed/60:.2f}min | "
    #       f"ETA: {est_remaining/60:.2f}min")
        
    #     if epoch%100 == 0:
    #         torch.save(agent.state_dict(), ckpt_file)
    #         print(f"✅ BC 模型权重已保存到 {ckpt_file}")


    N = states.shape[0]
    
    episode_length = 100
    
    episode_start = 62
    success_count = 21

    '''let's do joint overfitting first'''

    # try img first

    # or let's say, let oneshot traj be the mean output
    # then use gaussian distribution to output logprob
    # when doing online, 

    # camera warmup
    for _ in range(10):
        frame = env.get_obs()
        time.sleep(0.1)

    # run episode
    agent.eval()
    for ep in range(episode_length):
        env.reset()
        time.sleep(1)

        raw_pc = torch.tensor(np.zeros((2048,3)), dtype=torch.float32).to(agent.device)
        obs = {
                "pc": raw_pc,  # (2048,3)
                "q": env.robot.q.copy(),
                "t": 0
            }
        
        for time_step in tqdm(range(N)):
            frame = env.get_obs()

            if frame is None:
                continue
            
            rgb_img = frame["color"].copy()
            depth_img = frame["depth"].copy()
            raw_pc = frame["pointcloud"].copy()
            next_obs = {
                "pc": raw_pc,  # (2048,3)
                "q": env.robot.q,
                "t": time_step
            }  


            
            _agent_action  = agent.act(obs)    # np action ,delta+gripper
            if len(_agent_action.shape) != 1:
                _agent_action = _agent_action.squeeze(0)
            env.agent_action(_agent_action)  # agent action
            time.sleep(0.01)
        
            mouse_action, buttons = env.mouse.get_action()
            agent_done = env.get_agent_done(buttons)
            # left - sucess - 1; right - fail - 0
            if agent_done != 2:
                print("unsuccessful intervention,reset robot.")
                break
            obs = next_obs  
        if env.mouse.ask_success():
            agent_done = 1
        else:
            agent_done = 0
        total_ep = episode_start + ep
        reward = env.get_reward(agent_done)
            
            

        if reward >0.1:
            success_count += 1
            print(f"Episode {total_ep+1}/100 completed successfully!")

        writer.add_scalar('success_rate', success_count / (total_ep + 1), total_ep)
        cprint(f"Success Rate: {success_count} / {(total_ep + 1):.2f} ={success_count / (total_ep + 1):.2f} ", 'green')




'''
offline data
['obs_seq', 'next_obs_seq', 'joints_seq', 'actions_seq', 'rewards_seq', 'dones_seq']
(136, 2048, 3)
(136, 2048, 3)
(136, 7)
(136, 7)
(136,)
(136,)
'''
