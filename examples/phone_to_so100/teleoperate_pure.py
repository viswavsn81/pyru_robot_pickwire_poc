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

    # 3. Setup Math Solver
    motor_names = list(robot.bus.motors.keys())
    print(f"ℹ️ Math Solver using {len(motor_names)} joints: {motor_names}")

    kinematics = RobotKinematics(
        urdf_path="SO-ARM100/Simulation/SO100/so100.urdf",
        target_frame_name="jaw",
        joint_names=motor_names,
    )

    # 4. Pygame Window
    pygame.display.init()
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("CLICK HERE TO CONTROL ROBOT")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    print("\n🚀 TELEOP STARTED (SMART MODE)!")
    print("-------------------------------------")
    print("👉 STEP 1: Click the BLUE window.")
    print("👉 STEP 2: Move sticks to move the arm.")
    print("-------------------------------------\n")

    target_pos = None
    target_quat = None
    gripper_state = 0.0 
    
    # Track which IK signature works
    ik_mode = "unknown" 

    while True:
        t0 = time.perf_counter()
        pygame.event.pump()

        # --- A. READ SENSORS ---
        robot_obs = robot.get_observation()
        gamepad_data = teleop.get_action()
        raw_inputs = gamepad_data.get("gamepad.raw_inputs", {})

        # --- B. EXTRACT ALL JOINTS ---
        current_joints_deg = []
        try:
            for name in motor_names:
                key = f"{name}.pos"
                val = robot_obs[key]
                if hasattr(val, 'item'):
                    current_joints_deg.append(val.item())
                else:
                    current_joints_deg.append(val)
        except KeyError as e:
            print(f"❌ Critical Error: Missing key {e}!")
            break

        current_joints_rad = np.deg2rad(np.array(current_joints_deg))
        
        # --- C. INITIALIZE TARGET ---
        if target_pos is None:
            # Get Forward Kinematics
            current_ee = kinematics.forward_kinematics(current_joints_rad)
            
            # Sanitize Output
            if isinstance(current_ee, torch.Tensor):
                current_ee = current_ee.cpu().numpy()
            elif isinstance(current_ee, list):
                 current_ee = np.array(current_ee)

            # Extract Vector
            if isinstance(current_ee, tuple):
                 flat = np.concatenate([current_ee[0], current_ee[1]])
            else:
                 flat = current_ee.flatten()
            
            # Save Initial State
            target_pos = flat[:3].copy()
            target_quat = flat[3:].copy()
            
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
            
        # --- E. SOLVE IK (AUTO-DETECT MODE) ---
        target_joints_rad = None
        
        # 1. Prepare Full 7D Pose Vector (Pos + Quat)
        target_pose_7d = np.concatenate([target_pos, target_quat]).astype(np.float64)
        
        try:
            if ik_mode == "unknown" or ik_mode == "single_vector":
                # TRY 1: Pass single 7D vector (joints, pose)
                target_joints_rad = kinematics.inverse_kinematics(
                    current_joints_rad,
                    target_pose_7d
                )
                ik_mode = "single_vector" # It worked!
                
        except Exception:
            # If 1 failed, try 2
            if ik_mode == "unknown" or ik_mode == "split_args":
                try:
                    # TRY 2: Pass split arguments (joints, pos, quat)
                    # Ensure they are contiguous float64 arrays
                    target_joints_rad = kinematics.inverse_kinematics(
                        current_joints_rad,
                        np.ascontiguousarray(target_pos, dtype=np.float64),
                        np.ascontiguousarray(target_quat, dtype=np.float64)
                    )
                    ik_mode = "split_args" # It worked!
                except Exception as e2:
                    print(f"❌ IK Error: {e2}")
                    break

        if target_joints_rad is None:
             print("❌ Failed to solve IK with both methods.")
             break

        # Convert result to degrees
        if isinstance(target_joints_rad, torch.Tensor):
            target_joints_deg = np.rad2deg(target_joints_rad.cpu().numpy())
        else:
            target_joints_deg = np.rad2deg(target_joints_rad)
        
        # --- F. SEND COMMAND ---
        full_action = target_joints_deg.copy()
        
        # Overwrite Gripper
        if "gripper" in motor_names:
            idx = motor_names.index("gripper")
            full_action[idx] = 45 + (gripper_state * 45)

        robot.send_action(torch.tensor(full_action, dtype=torch.float32))

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
