#!/usr/bin/env python

import time
import numpy as np
import torch
import pygame
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

def main():
    print("🔌 Connecting to Robot...")
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ NO CONTROLLER FOUND!")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Window setup
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("FULL 6-AXIS CONTROL")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    print("\n🚀 FULL CONTROL ACTIVE!")
    print("-------------------------------------")
    print("🕹️  L-STICK:      Base & Shoulder")
    print("🕹️  R-STICK (Y):  Elbow")
    print("🕹️  R-STICK (X):  Wrist Flex (Joint 4)")
    print("🔘 LB / RB:      Wrist Roll (Joint 5)")
    print("🔘 A / B:        Gripper")
    print("-------------------------------------\n")

    # Lock Initial Positions
    motor_names = list(robot.bus.motors.keys())
    current_obs = robot.get_observation()
    target_joints = {}
    
    for name in motor_names:
        key = f"{name}.pos"
        val = current_obs[key]
        if hasattr(val, "item"): val = val.item()
        target_joints[name] = val

    print("✅ Ready. Click the Blue Window to Start.")

    while True:
        t0 = time.perf_counter()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # --- READ INPUTS ---
        # Axes
        ax0 = joystick.get_axis(0) # LS Left/Right
        ax1 = joystick.get_axis(1) # LS Up/Down
        ax3 = joystick.get_axis(3) # RS Left/Right (Check your scanner if this doesn't work!)
        ax4 = joystick.get_axis(4) # RS Up/Down

        # Buttons (LB=4, RB=5 usually)
        # Note: Button IDs can vary. If LB/RB don't work, try buttons 0-10 in scanner.
        lb = joystick.get_button(4) 
        rb = joystick.get_button(5)
        btn_a = joystick.get_button(0)
        btn_b = joystick.get_button(1)

        speed = 1.2 # Speed multiplier

        # --- JOINT 1: Base (Shoulder Pan) ---
        if abs(ax0) > 0.1:
             # We keep the inverted direction you liked
             target_joints["shoulder_pan"] += ax0 * speed

        # --- JOINT 2: Shoulder (Shoulder Lift) ---
        if abs(ax1) > 0.1:
             target_joints["shoulder_lift"] -= ax1 * speed

        # --- JOINT 3: Elbow (Elbow Flex) ---
        if abs(ax4) > 0.1:
             target_joints["elbow_flex"] -= ax4 * speed

        # --- JOINT 4: Wrist Flex (Right Stick X) ---
        if abs(ax3) > 0.1:
             # Try -= or += to swap direction if needed
             target_joints["wrist_flex"] -= ax3 * speed

        # --- JOINT 5: Wrist Roll (Bumpers) ---
        if lb:
             target_joints["wrist_roll"] -= speed * 1.5 # Roll Left
        if rb:
             target_joints["wrist_roll"] += speed * 1.5 # Roll Right

        # --- JOINT 6: Gripper ---
        if btn_a: target_joints["gripper"] = -45
        if btn_b: target_joints["gripper"] = 90

        # --- SAFETY CLAMPS ---
        for name in target_joints:
            if name != "gripper":
                # Most joints act weird beyond +/- 175 degrees
                target_joints[name] = np.clip(target_joints[name], -175, 175)

        # --- SEND ---
        action_dict = {}
        for name, angle in target_joints.items():
            action_dict[f"{name}.pos"] = torch.tensor([angle], dtype=torch.float32)

        robot.send_action(action_dict)
        
        # Debug Print (Overwrite line)
        print(f"\rJ4(Flex): {target_joints['wrist_flex']:.1f} | J5(Roll): {target_joints['wrist_roll']:.1f}   ", end="")

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
