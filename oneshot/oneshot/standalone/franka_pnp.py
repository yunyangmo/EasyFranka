import os,sys,time,copy
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import datetime
import threading
from collections import deque
from typing import Tuple
import requests
import cprint
import torch 
import numpy as np
import cv2
import pyspacemouse
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from oneshot.utils import Franka,RSCapture,SpaceMouseExpert
from oneshot.utils.operation import transform_camera_to_marker,interpolate_se3_euler
from oneshot.utils.camera import voxel_downsampling,furthest_point_sampling,PointNetEncoderXYZ,fast_furthest_point_sampling

from ultralytics import YOLO
from oneshot.envs import FrankaEnv

 
if __name__ == "__main__":
    env = FrankaEnv()  
    time.sleep(1)

    obs_seq = []
    next_obs_seq = []
    joints_seq= []
    actions_seq = []    # change into using delta action + gripper
    rewards_seq = []
    dones_seq = []


    env.reset()

    while True:
        
        frame = env.get_obs()


        rgb_img = frame["color"].copy()
        depth_img = frame["depth"].copy()
        next_obs = frame["pointcloud"].copy()
        obs = frame["pointnet"]
        mouse_action, buttons = env.mouse.get_action()
        print(f'mouse_action : {mouse_action}')
        print(f'buttons : {buttons}')
        print(f'buttons: {buttons}')
        env.move_robot(mouse_action, buttons)

        # 8 dimension,  ee pose and  gripper pos
        # 7 dims delta-xyz, euler, gripper
        action = env.get_action(if_delta=True) 

        done = env.get_done(buttons)
        reward = env.get_reward(rgb_img, done)
        
        cv2.imshow('image',rgb_img)


        joints_seq.append(env.robot.q.copy())
        obs_seq.append(obs.copy())
        next_obs_seq.append(next_obs.copy())
        actions_seq.append(action.copy())
        rewards_seq.append(reward)
        dones_seq.append(done)

        obs = next_obs.copy()

        if done == 1:
            break
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break 
        time.sleep(0.02)
    
        

