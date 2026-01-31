#!/usr/bin/env python

import time
import numpy as np
import torch
import pygame
from pathlib import Path
from dataclasses import dataclass

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop

FPS = 30

@dataclass
class GamepadTeleopConfig:
    type: str = "Xbox Wireless Controller"
    id: str = "xbox_controller"  
    streaming_port: int = 9999
    max_speed: float = 1.0
    fps: int = 30
    calibration_dir: Path = None
    robot_type: str = "so100"
    use_gripper: bool = True
    
    def __post_init__(self):
        if self.calibration_dir is None:
            self.calibration_dir = Path(".cache/calibration")

# --- HELPER: ROTATION MATH ---
def make_4x4_matrix(pos, quat):
    x, y, z = pos
    # FIX: Ensure we only take the first 4 elements if quat is longer
    qx, qy, qz, qw = quat[:4]
    
    # Formula to convert Quaternion to Rotation Matrix
    r00 = 1 - 2*qy*qy - 2*qz*qz
    r01 = 2*qx*qy - 2*qz*qw
    r02 = 2*qx*qz + 2*qy*qw
    
    r10 = 2*qx*qy + 2*qz*qw
    r11 = 1 - 2*qx*qx - 2*qz*qz
    r12 = 2*qy*qz - 2*qx*qw
    
    r20 = 2*qx*qz - 2*qy*qw
    r21 = 2*qy*qz + 2*qx*qw
    r22 = 1 - 2*qx*qx - 2*qy*qy
    
    # Assemble the 4x4 Matrix
    matrix = np.array([
        [r00, r01, r02, x],
        [r10, r11, r12, y],
        [r20, r21, r22, z],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    return matrix

def main():
    # 1. Initialize Robot
    print("🔌 Connecting to Robot...")
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    # 2. Initialize Xbox
    print("🎮 Connecting to Xbox Controller...")
    teleop = GamepadTeleop(GamepadTeleopConfig())
    teleop.connect()
    print("✅ Controller Connected!")

    # 3. Setup Math Solver (5-Axis Only)
    arm_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    motor_names = list(robot.bus.motors.keys()) 
    
    print(f"ℹ️ Math Solver Configured for: {arm_joints}")

    kinematics = RobotKinematics(
        urdf_path="SO-ARM100/Simulation/SO100/so100.urdf",
        target_frame_name="jaw",
        joint_names=arm_joints,
    )

    # 4. Pygame Window
    pygame.display.init()
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("CLICK HERE TO CONTROL ROBOT")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    print("\n🚀 TELEOP STARTED (MATRIX MODE FIXED)!")
    print("-------------------------------------")
    print("👉 STEP 1: Click the BLUE window.")
    print("👉 STEP 2: Move sticks to move the arm.")
    print("-------------------------------------\n")

    target_pos = None
    target_quat = None
    gripper_state = 0.0 

    while True:
        t0 = time.perf_counter()
        pygame.event.pump()

        # --- A. READ SENSORS ---
        robot_obs = robot.get_observation()
        gamepad_data = teleop.get_action()
        raw_inputs = gamepad_data.get("gamepad.raw_inputs", {})

        # --- B. EXTRACT 5 ARM JOINTS ---
        current_arm_deg = []
        try:
            for name in arm_joints:
                key = f"{name}.pos"
                val = robot_obs[key]
                if hasattr(val, 'item'):
                    current_arm_deg.append(val.item())
                else:
                    current_arm_deg.append(val)
        except KeyError as e:
            print(f"❌ Critical Error: Missing key {e}!")
            break

        current_arm_rad = np.deg2rad(np.array(current_arm_deg))
        
        # --- C. INITIALIZE TARGET ---
        if target_pos is None:
            current_ee = kinematics.forward_kinematics(current_arm_rad)
            
            # Sanitize Output
            if isinstance(current_ee, torch.Tensor):
                current_ee = current_ee.cpu().numpy()
            elif isinstance(current_ee, list):
                 current_ee = np.array(current_ee)

            if isinstance(current_ee, tuple):
                 flat = np.concatenate([current_ee[0], current_ee[1]])
            else:
                 flat = current_ee.flatten()
            
            target_pos = flat[:3].copy()
            # FIX: Ensure we only slice exactly 4 values for quaternion
            target_quat = flat[3:7].copy()
            print(f"✅ Target initialized at: {target_pos}")

        # --- D. CALCULATE MOTION ---
        scale = 0.005 
        
        dx = -raw_inputs.get("axis1", 0.0) * scale 
        dy = -raw_inputs.get("axis0", 0.0) * scale 
        dz = -raw_inputs.get("axis4", 0.0) * scale 

        target_pos[0] += dx
        target_pos[1] += dy
        target_pos[2] += dz

        if raw_inputs.get("btn0", 0): gripper_state = -1.0
        elif raw_inputs.get("btn1", 0): gripper_state = 1.0
            
        # --- E. SOLVE IK (MATRIX MODE) ---
        target_matrix = make_4x4_matrix(target_pos, target_quat)
        
        target_joints_rad = kinematics.inverse_kinematics(
            current_arm_rad,
            target_matrix
        )

        if isinstance(target_joints_rad, torch.Tensor):
            target_joints_deg = np.rad2deg(target_joints_rad.cpu().numpy())
        else:
            target_joints_deg = np.rad2deg(target_joints_rad)
        
        # --- F. SEND COMMAND (6-Axis) ---
        full_action = []
        
        for name in motor_names:
            if name == "gripper":
                angle = 0 + (gripper_state * 45) + 45 
                full_action.append(angle)
            elif name in arm_joints:
                idx = arm_joints.index(name)
                full_action.append(target_joints_deg[idx])
            else:
                full_action.append(0.0)

        robot.send_action(torch.tensor(full_action, dtype=torch.float32))

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
