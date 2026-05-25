import os,sys,time,copy

import datetime
import threading
from collections import deque
from typing import Tuple
import requests

import numpy as np

import pyspacemouse
import tkinter as tk
from tkinter import messagebox


class SpaceMouseExpert:

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]
        
    def ask_success(self) -> bool:

        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        result = messagebox.askyesno("Task Result", "任务是否成功？")
        root.destroy()
        return result