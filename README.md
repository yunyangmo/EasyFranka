# EasyFranka

EasyFranka is a lightweight workflow repo for deploying Franka teleoperation and data collection.

This project is built on top of and modified from the Franka workflow used in [`serl_franka_controllers`](https://github.com/rail-berkeley/serl_franka_controllers.git) and [`serl`](https://github.com/rail-berkeley/serl.git). Users can first deploy the Franka controller stack locally, and then use this repo to run a simplified teleoperation and data collection pipeline.

---

## Installation

### 1. Install `serl_franka_controllers`

First, install [`serl_franka_controllers`](https://github.com/rail-berkeley/serl_franka_controllers.git) on a computer that has both `libfranka` and `franka_ros` installed.

Please follow the official Franka requirements and setup instructions:

https://frankaemika.github.io/docs/requirements.html

A typical setup should have:

- a working Franka robot connection;
- `libfranka` installed;
- `franka_ros` installed;
- network access to the Franka robot;
- a properly configured real-time control environment.

---

### 2. Install `serl_robot_infra`

This package is used to start a web server that bridges the connection between the Franka robot and command-line requests such as `curl`.

Go to the `serl_robot_infra` directory:

```bash
conda create -n serl python=3.10
conda activate serl

cd /path/to/serl_robot_infra
pip install -e .
```

Then start the Franka server:

```bash
python serl_robot_infra/robot_servers/franka_server.py \
    --gripper_type=<Robotiq|Franka|None> \
    --robot_ip=<robot_IP>
```

For example, if you are using a Franka gripper:

```bash
python serl_robot_infra/robot_servers/franka_server.py \
    --gripper_type=Franka \
    --robot_ip=172.16.0.2
```

---

### 3. Install this repo

Install the `oneshot` package in editable mode:

```bash
conda create -n oneshotRL python=3.10
conda activate oneshotRL

cd EasyFranka
pip install -e ./oneshot
```

This allows you to modify the source code locally while keeping the package importable from your Python environment.

---

### 4. Check SpaceMouse and RealSense camera devices

Before running teleoperation or data collection, make sure your SpaceMouse and RealSense camera dependencies are correctly installed and connected.

You can check connected devices using:

```bash
python oneshot/standalone/check_devices.py
```

This script can be used to verify device availability and obtain RealSense camera serial numbers.

---

### 5. Run simple teleoperation

From the `EasyFranka` root directory, run:

```bash
python oneshot/standalone/franka_pnp.py
```

If you are already inside the `oneshot` directory, run:

```bash
python standalone/franka_pnp.py
```

---

## Project Structure

```text
EasyFranka/
├── README.md
├── .gitignore
└── oneshot/
    ├── agents/
    ├── algos/
    ├── config/
    ├── envs/
    ├── standalone/
    │   ├── check_devices.py
    │   └── franka_pnp.py
    └── utils/
```

---

## Basic Workflow

The recommended workflow is:

1. Set up the Franka control environment with `libfranka`, `franka_ros`, and `serl_franka_controllers`.
2. Install and launch `serl_robot_infra` to start the Franka robot server.
3. Install the `oneshot` package from this repository.
4. Check SpaceMouse and RealSense camera connections.
5. Run teleoperation and data collection scripts from `oneshot/standalone`.

---

## Notes

- It is recommended to run `serl_robot_infra` and this project on the same computer connected to the Franka robot.
- Make sure the robot IP address is reachable before starting the Franka server.
- Make sure the SpaceMouse and RealSense cameras are connected before launching teleoperation or data collection.
- For RealSense multi-camera setups, use `check_devices.py` to identify and record the correct camera serial numbers.
- You can also refer to the original installation instructions from [`serl`](https://github.com/rail-berkeley/serl.git), and then skip to Step 3 for installing this repo.

---

## Acknowledgement

This repository is based on and modified from code and workflow components originally developed in the SERL Franka ecosystem, including:

- [`serl_franka_controllers`](https://github.com/rail-berkeley/serl_franka_controllers.git)
- [`serl`](https://github.com/rail-berkeley/serl.git)
- the Franka software stack: https://frankaemika.github.io/docs/requirements.html

This project is an independent modified version and is not officially endorsed by, affiliated with, or maintained by the original authors or organizations.

---

## License

This project is licensed under the Apache License 2.0.

Portions of this project are modified from code originally copyrighted by Jianlan Luo, Zheyuan Hu, Charles Xu, and You-Liang Tan.

Modifications in this repository are copyrighted by Yunyang Mo.

Please refer to the `LICENSE` file for details.