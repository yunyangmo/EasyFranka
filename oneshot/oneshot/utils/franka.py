import os,sys,time,copy
import open3d
import json
from tqdm import tqdm
import pickle as pkl
import datetime
import threading
from collections import deque
from typing import Tuple
import requests
import torch
import numpy as np
import cv2
import pyspacemouse
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


RESET_POSE = [0.59, 0.0777, 0.31378, 3.1099675, 0.0146619, -0.0078615]

# RESET_JOINTS = [    0.12081,    -0.84518,    0.043024,     -2.3642,     0.01638,      1.4503,     -2.3614]
RESET_JOINTS = [  0.0026492,     0.39198,    0.055768,     -1.5627,   -0.039242,      2.0566,     -2.2338]

# RANGE_LOW = [-0.1, -0.1, -0.3, -0.33, -0.33, -0.3333]
# RANGE_LOW = [-0.4, -0.3, -0.3, -3.14, -3.14, -3.14]
# RANGE_HIGH = [0.3, 0.3, 0.3, 3.14, 3.14, 3.14]

RANGE_LOW = [-0.5, -0.4, -0.4, -3.14, -3.14, -3.14]
RANGE_HIGH = [0.4, 0.4, 0.4, 3.14, 3.14, 3.14]
def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")
def euler_2_quat(xyz):
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0], 
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements

class Franka:
    def __init__(self, action_scales=[0.04, 0.1, 20]):


        # get state
        self.url = 'http://192.168.1.11:5000/'

        self.resetpos = np.concatenate([RESET_POSE[:3], euler_2_quat(RESET_POSE[3:])])
        self.currpos = self.resetpos.copy() # [x,y,z,qx,qy,qz,qw]
        
        self.cur_joints = np.zeros((7),)

        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.curr_gripper_pos = 0 # 0-200

        self._update_currpos()

        self.xyz_low = (np.array(RESET_POSE) + np.array(RANGE_LOW))[:3]
        self.xyz_high = (np.array(RESET_POSE) + np.array(RANGE_HIGH))[:3]
        self.euler_low = (np.array(RESET_POSE) + np.array(RANGE_LOW))[3:]
        self.euler_high = (np.array(RESET_POSE) + np.array(RANGE_HIGH))[3:]

        self.action_scales = action_scales

        

    def clip_safety_box(self, pose):
    
        pose[:3] = np.clip(pose[:3], self.xyz_low, self.xyz_high)
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
                np.clip(
                    np.abs(euler[0]),
                    self.euler_low[0],
                    self.euler_high[0],
                )
        )
        euler[1:] = np.clip(
                euler[1:], self.euler_low[1:], self.euler_high[1:]
            )
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose

    def clip_safety_euler_box(self, pose):
    
        pose[:3] = np.clip(pose[:3], self.xyz_low, self.xyz_high)
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
                np.clip(
                    np.abs(euler[0]),
                    self.euler_low[0],
                    self.euler_high[0],
                )
        )
        euler[1:] = np.clip(
                euler[1:], self.euler_low[1:], self.euler_high[1:]
            )

        new_pos = np.zeros(6, dtype=np.float32)
        new_pos[:3] = pose[:3]
        new_pos[3:] = euler

        return new_pos

    def agent_move(self, xyz_delta, euler_delta, gripper):
        self.nextpos = self.currpos.copy()
        xyz_delta = xyz_delta if isinstance(xyz_delta, np.ndarray) else xyz_delta.numpy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta
        euler_delta = euler_delta if isinstance(euler_delta, np.ndarray) else euler_delta.numpy()
        next_euler_pos = (
            R.from_euler('xyz', (euler_delta)) *
            R.from_quat(self.currpos[3:])
            ).as_quat()
        
        self.nextpos[3:] = next_euler_pos
        
        self.delta_xyz = np.array(xyz_delta)*self.action_scales[0]
        self.delta_euler = np.array(euler_delta)*self.action_scales[1]
        gripper = gripper.detach().cpu().numpy() if isinstance(gripper, torch.Tensor) else gripper
        arr = 1.*gripper
        self.next_gripper_pos = arr 

        data = {"gripper_pos": arr}
        headers = {"Content-Type":"application/json"}
        
        requests.post(self.url + "move_gripper", headers= headers, json=data)
        self._send_pos_command(self.clip_safety_box(self.nextpos))
    
        self._update_currpos()

    def agent_absolute_move(self, xyz_absolute, euler_absolute, gripper):
        self.nextpos = self.currpos.copy()
        
        # 处理位置输入（绝对坐标）
        xyz_absolute = xyz_absolute if isinstance(xyz_absolute, np.ndarray) else xyz_absolute.numpy()
        self.nextpos[:3] = xyz_absolute  # 直接使用绝对坐标
        
        # 处理姿态输入（绝对欧拉角）
        euler_absolute = euler_absolute if isinstance(euler_absolute, np.ndarray) else euler_absolute.numpy()
        
        # 将绝对欧拉角转换为四元数
        next_euler_pos = R.from_euler('xyz', euler_absolute).as_quat()
        self.nextpos[3:] = next_euler_pos
        
        # 计算实际执行的delta值（用于记录或监控）
        self.delta_xyz = (xyz_absolute - self.currpos[:3]) * self.action_scales[0]
        self.delta_euler = (euler_absolute - R.from_quat(self.currpos[3:]).as_euler('xyz')) * self.action_scales[1]
        
        # 处理夹爪控制（保持不变）
        gripper = gripper.detach().cpu().numpy() if isinstance(gripper, torch.Tensor) else gripper
        arr = 1. * gripper
        self.next_gripper_pos = arr 

        data = {"gripper_pos": arr}
        headers = {"Content-Type": "application/json"}
        
        requests.post(self.url + "move_gripper", headers=headers, json=data)
        self._send_pos_command(self.clip_safety_box(self.nextpos))
        # self._send_pos_command(self.nextpos)
        
        self._update_currpos()

    def move(self, xyzrpy, gripper_flag):
        xyz_delta = xyzrpy[:3]
        euler_delta = xyzrpy[3:]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + self.action_scales[0] * np.array(xyz_delta)
        
        self.delta_xyz = np.array(xyz_delta)*self.action_scales[0]
        self.delta_euler = np.array(euler_delta)*self.action_scales[1]
        
        
        next_euler_pos = (
            R.from_euler('xyz', (np.array(euler_delta)*self.action_scales[1])) *
            R.from_quat(self.currpos[3:])
            ).as_quat()
        self.nextpos[3:] = next_euler_pos

        
        if gripper_flag == 0:
            gripper_action = 0
        elif gripper_flag == 1:
            gripper_action = self.action_scales[-1]
        elif gripper_flag == -1:
            gripper_action = -self.action_scales[-1]
        
        ##
        self._send_gripper_command(gripper_action,mode="continuous")
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self._update_currpos()

    def reset(self):
        reset_gripper = 222
        for _ in range(4):
            self._send_gripper_command(reset_gripper,mode="continuous")

        reset_target = RESET_JOINTS
        data = {"target": reset_target}
        
        response = requests.post(
            self.url + "set_joint_target", 
            headers={"Content-Type": "application/json"},
            json=data
            )
        
        if response.status_code == 200:
            print("Robot reset to joint target successfully.")
            print(response.json())
        else:
            print(f"Failed to reset robot: {response.text}")
            # print(response.json())
        time.sleep(3) 

        # this code might take a few seconds
        resp = requests.post(self.url +"jointreset")
        print(resp.text)

        
        time.sleep(1)
        for _ in range(4):
            self._send_gripper_command(reset_gripper,mode="continuous")
        time.sleep(1)
        self._update_currpos()


    def _send_gripper_command(self, pos: float, mode="binary"):
        if mode == "binary":
            if (
                pos <= 0
                and self.gripper_binary_state == 0
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 1
                return True
            elif (
                pos >= 0
                and self.gripper_binary_state == 1
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 0
                return True
            else:  # do nothing to the gripper
                return False
        elif mode == "continuous":
            # Store current gripper position in a local variable to avoid concurrent updates
            current_pos = self.curr_gripper_pos.copy()

            next_gripper_pos = 2550*current_pos + pos   
            self.next_gripper_pos = next_gripper_pos
            # print(next_gripper_pos)

            arr = next_gripper_pos
            data = {"gripper_pos": arr}
            headers = {"Content-Type":"application/json"}
            # print(data)
            requests.post(self.url + "move_gripper", headers= headers, json=data)

            return True 
    
    def _send_pos_command(self, pos: np.ndarray):
    
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}

        
        

        requests.post(self.url + "pose", json=data) 

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")      

    def _update_currpos(self):
        ps = requests.post(self.url + 'getstate').json()
        self.currpos[:] = np.array(ps["pose"])

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])
