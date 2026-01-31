#!/usr/bin/env python

import time
import numpy as np
import torch
import pygame
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

def main():
    # 1. Initialize Robot
    print("🔌 Connecting to Robot...")
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    # 2. Initialize Pygame
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ NO CONTROLLER FOUND!")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 Controller Found: {joystick.get_name()}")

    # 3. Setup Window
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("CLICK ME TO CONTROL")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    print("\n🚀 DIRECT CONTROL ACTIVE (INVERTED PAN)!")
    print("-------------------------------------")
    print("👉 STEP 1: Click the BLUE window.")
    print("👉 STEP 2: Move sticks.")
    print("-------------------------------------\n")

    # 4. Lock Initial Positions
    motor_names = list(robot.bus.motors.keys())
    current_obs = robot.get_observation()
    target_joints = {}
    
    for name in motor_names:
        key = f"{name}.pos"
        val = current_obs[key]
        if hasattr(val, "item"): val = val.item()
        target_joints[name] = val

    print("✅ Positions Locked. Ready.")

    while True:
        t0 = time.perf_counter()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # READ AXES
        axis_0 = joystick.get_axis(0) # Base (Left/Right)
        axis_1 = joystick.get_axis(1) # Shoulder (Up/Down)
        axis_4 = joystick.get_axis(4) # Elbow (Up/Down) - Might be axis 3 on some pads
        
        print(f"\raxes: {axis_0:.2f} {axis_1:.2f} {axis_4:.2f} | ", end="")

        speed = 1.0 

        # --- FIX: FLIPPED SIGN FOR BASE ---
        if abs(axis_0) > 0.1:
            if "shoulder_pan" in target_joints: 
                # Changed from -= to += to flip direction
                target_joints["shoulder_pan"] += axis_0 * speed
            
        if abs(axis_1) > 0.1:
            if "shoulder_lift" in target_joints: target_joints["shoulder_lift"] -= axis_1 * speed
            
        if abs(axis_4) > 0.1:
            if "elbow_flex" in target_joints: target_joints["elbow_flex"] -= axis_4 * speed

        # BUTTONS
        if joystick.get_button(0): # A
             if "gripper" in target_joints: target_joints["gripper"] = -45
        if joystick.get_button(1): # B
             if "gripper" in target_joints: target_joints["gripper"] = 90

        # SAFETY CLAMPS
        for name in target_joints:
            if name != "gripper":
                target_joints[name] = np.clip(target_joints[name], -175, 175)

        # SEND
        action_dict = {}
        for name, angle in target_joints.items():
            action_dict[f"{name}.pos"] = torch.tensor([angle], dtype=torch.float32)

        robot.send_action(action_dict)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
