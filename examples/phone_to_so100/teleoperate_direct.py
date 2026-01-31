#!/usr/bin/env python

import time
import numpy as np
import torch
import pygame
from dataclasses import dataclass
from pathlib import Path

# We only need the Robot and Gamepad. NO MATH SOLVER.
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.utils.robot_utils import precise_sleep

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

    # 3. Identify Motors
    # Expected: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    motor_names = list(robot.bus.motors.keys())
    print(f"ℹ️ Motors found: {motor_names}")

    # 4. Pygame Window
    pygame.display.init()
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("DIRECT JOINT CONTROL")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    print("\n🚀 TELEOP STARTED (DIRECT DRIVE)!")
    print("-------------------------------------")
    print("🕹️  LEFT STICK:  Base (Pan) / Shoulder (Lift)")
    print("🕹️  RIGHT STICK: Elbow (Flex)")
    print("🔫 TRIGGERS:    Wrist (Flex)")
    print("🔘 A/B:         Gripper Open/Close")
    print("-------------------------------------\n")

    # Capture initial positions so the robot doesn't snap
    current_obs = robot.get_observation()
    target_joints = {}
    
    for name in motor_names:
        key = f"{name}.pos"
        # Store as simple float
        val = current_obs[key]
        if hasattr(val, "item"): val = val.item()
        target_joints[name] = val

    print("✅ Initial Positions Locked. Moving...")

    while True:
        t0 = time.perf_counter()
        pygame.event.pump()

        gamepad_data = teleop.get_action()
        raw = gamepad_data.get("gamepad.raw_inputs", {})

        # --- MAPPING: STICK -> MOTOR ---
        speed = 1.5 # Degrees per frame
        
        # 1. Base (Shoulder Pan) - Left Stick X
        if "shoulder_pan" in target_joints:
            target_joints["shoulder_pan"] -= raw.get("axis0", 0.0) * speed

        # 2. Shoulder (Shoulder Lift) - Left Stick Y
        if "shoulder_lift" in target_joints:
            target_joints["shoulder_lift"] -= raw.get("axis1", 0.0) * speed

        # 3. Elbow (Elbow Flex) - Right Stick Y
        if "elbow_flex" in target_joints:
            target_joints["elbow_flex"] -= raw.get("axis4", 0.0) * speed

        # 4. Wrist Flex - Triggers (Axis 2=LT, Axis 5=RT on some controllers)
        # Often Triggers are -1 to 1. 
        # Let's use Right Stick X as backup for Wrist
        if "wrist_flex" in target_joints:
             target_joints["wrist_flex"] -= raw.get("axis3", 0.0) * speed

        # 5. Wrist Roll - Bumpers? Let's skip for now or use D-Pad if available
        # Simplified: Stick to main 3 axes first.

        # 6. Gripper - A/B Buttons
        if "gripper" in target_joints:
            if raw.get("btn0", 0): # A
                target_joints["gripper"] = -45.0 # Close
            elif raw.get("btn1", 0): # B
                target_joints["gripper"] = 90.0 # Open

        # --- SAFETY CLAMPS ---
        # Don't let motors spin forever
        for name in target_joints:
            if name != "gripper":
                target_joints[name] = np.clip(target_joints[name], -175, 175)

        # --- SEND COMMAND ---
        action_dict = {}
        for name, angle in target_joints.items():
            action_dict[f"{name}.pos"] = torch.tensor([angle], dtype=torch.float32)

        robot.send_action(action_dict)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
