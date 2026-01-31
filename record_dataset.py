#!/usr/bin/env python

import os
import time
import json
import cv2
import torch
import numpy as np
import pygame
from pathlib import Path
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep

# Configuration
FPS = 30
DATASET_ROOT = "dataset"

def ensure_episode_dir(root):
    """Creates a new episode directory (e.g., episode_001) and returns it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # Find next episode index
    existing_eps = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    max_idx = 0
    for p in existing_eps:
        try:
            idx = int(p.name.split("_")[1])
            max_idx = max(max_idx, idx)
        except:
            pass
    
    new_ep_idx = max_idx + 1
    new_ep_dir = root / f"episode_{new_ep_idx:03d}"
    new_ep_dir.mkdir(parents=True, exist_ok=True)
    return new_ep_dir

def main():
    # -----------------------------
    # 1. SETUP ROBOT & CAMERAS
    # -----------------------------
    print("🔌 Connecting to Robot...")
    # Logic from smoke_test_v2.py (which came from teleoperate_full_control.py)
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    print("🔌 Connecting to Cameras...")
    # Laptop: 2, Wrist: 0
    # Use MJPEG to reduce USB bandwidth (fixes flickering on high-res dual streams)
    laptop_config = OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480, fourcc="MJPG")
    wrist_config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480, fourcc="MJPG")
    
    laptop_cam = OpenCVCamera(laptop_config)
    wrist_cam = OpenCVCamera(wrist_config)
    
    try:
        laptop_cam.connect()
        wrist_cam.connect()
        # Start async threads immediately
        laptop_cam.async_read()
        wrist_cam.async_read()
        print("✅ Cameras Connected & Threads Started!")
        time.sleep(1) # Warmup
    except Exception as e:
        print(f"❌ Camera connection failed: {e}")
        return

    # -----------------------------
    # 2. SETUP PYGAME CONTROLLER
    # -----------------------------
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ NO CONTROLLER FOUND! Please connect Xbox controller.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 Controller connected: {joystick.get_name()}")

    # -----------------------------
    # 3. SETUP VISUALIZATION
    # -----------------------------
    # Use Pygame for display instead of OpenCV (avoids highgui errors)
    WINDOW_W, WINDOW_H = 1280, 480 # Two 640x480 images side-by-side
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("SO-100 Recorder - Press Y to Record, X to Exit")

    # -----------------------------
    # 4. INITIALIZE ROBOT STATE
    # -----------------------------
    motor_names = list(robot.bus.motors.keys())
    current_obs = robot.get_observation()
    target_joints = {}
    
    for name in motor_names:
        key = f"{name}.pos"
        val = current_obs[key]
        if hasattr(val, "item"): val = val.item()
        target_joints[name] = val

    # -----------------------------
    # 5. CONTROL & RECORDING LOOP
    # -----------------------------
    print("\n🚀 READY! Controls:")
    print("  [Start] / [Y]: Toggle Recording")
    print("  [Back] / [X]:  Exit")
    print("  Sticks/Bumpers/Buttons: Move Robot\n")

    recording = False
    current_ep_dir = None
    frame_idx = 0
    
    running = True
    while running:
        t0 = time.perf_counter()
        
        # Process Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Button Downs for Toggles
            if event.type == pygame.JOYBUTTONDOWN:
                # Exit (Back or X)
                if event.button == 6 or event.button == 2:
                    print("🛑 Exit requested.")
                    running = False
                
                # Toggle Recording (Start or Y)
                if event.button == 7 or event.button == 3:
                    recording = not recording
                    if recording:
                        current_ep_dir = ensure_episode_dir(DATASET_ROOT)
                        frame_idx = 0
                        print(f"🔴 RECORDING STARTED: {current_ep_dir}")
                    else:
                        print("⚪ RECORDING STOPPED")

        # --- READ CONTROLS (Teleoperation) ---
        ax0 = joystick.get_axis(0) # LS Left/Right
        ax1 = joystick.get_axis(1) # LS Up/Down
        ax3 = joystick.get_axis(3) # RS Left/Right
        ax4 = joystick.get_axis(4) # RS Up/Down
        lb = joystick.get_button(4) 
        rb = joystick.get_button(5)
        btn_a = joystick.get_button(0)
        btn_b = joystick.get_button(1)

        speed = 0.3
        if abs(ax0) > 0.1: target_joints["shoulder_pan"] += ax0 * speed
        if abs(ax1) > 0.1: target_joints["shoulder_lift"] -= ax1 * speed
        if abs(ax4) > 0.1: target_joints["elbow_flex"] -= ax4 * speed
        if abs(ax3) > 0.1: target_joints["wrist_flex"] -= ax3 * speed
        if lb: target_joints["wrist_roll"] -= speed * 1.5
        if rb: target_joints["wrist_roll"] += speed * 1.5
        if btn_a: target_joints["gripper"] = -45
        if btn_b: target_joints["gripper"] = 90

        # Clamps
        for name in target_joints:
            if name != "gripper":
                target_joints[name] = np.clip(target_joints[name], -175, 175)

        # Send Action
        action_dict = {}
        for name, angle in target_joints.items():
            action_dict[f"{name}.pos"] = torch.tensor([angle], dtype=torch.float32)
        robot.send_action(action_dict)

        # --- READ SENSORS ---
        # Decoupled Read: Access latest frame directly without blocking
        # This prevents 30Hz loop from stalling if camera is slow/jittery
        with laptop_cam.frame_lock:
            laptop_img = laptop_cam.latest_frame
        with wrist_cam.frame_lock:
            wrist_img = wrist_cam.latest_frame
            
        robot_obs = robot.get_observation() 

        # --- VISUALIZATION ---
        if laptop_img is not None and wrist_img is not None:
            # Convert LeRobot RGB (default) to Pygame-friendly format
            # Combine horizontally
            viz_img = np.hstack((laptop_img, wrist_img))
            
            # Pygame Surface
            # Transpose from (H, W, C) to (W, H, C)
            viz_surf = pygame.surfarray.make_surface(viz_img.swapaxes(0, 1))
            
            screen.blit(viz_surf, (0, 0))
            
            # Overlay Recording Status
            if recording:
                pygame.draw.circle(screen, (255, 0, 0), (30, 30), 10) # Red Circle
                font = pygame.font.SysFont(None, 36)
                text = font.render(f"REC Frame: {frame_idx}", True, (255, 0, 0))
                screen.blit(text, (50, 20))
            
            pygame.display.flip()

            # --- RECORDING ---
            if recording and current_ep_dir:
                # Save Images (OpenCV needs BGR)
                laptop_bgr = cv2.cvtColor(laptop_img, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(str(current_ep_dir / f"laptop_{frame_idx:06d}.jpg"), laptop_bgr)
                cv2.imwrite(str(current_ep_dir / f"wrist_{frame_idx:06d}.jpg"), wrist_bgr)
                
                # Save JSON
                serializable_obs = {}
                for k, v in robot_obs.items():
                    if hasattr(v, "item"):
                        serializable_obs[k] = v.item()
                    elif isinstance(v, (np.ndarray, list)):
                        serializable_obs[k] = v.tolist() if isinstance(v, np.ndarray) else v
                    else:
                        serializable_obs[k] = v
                
                data_point = {
                    "frame_index": frame_idx,
                    "timestamp": time.time(),
                    "observation": serializable_obs,
                    "action": target_joints
                }
                
                with open(current_ep_dir / f"frame_{frame_idx:06d}.json", "w") as f:
                    json.dump(data_point, f)
                
                frame_idx += 1

        # Timing
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    # Cleanup
    print("🧹 Closing connections...")
    try:
        robot.disconnect()
    except Exception as e:
        print(f"⚠️ Robot disconnect warning (safe to ignore): {e}")

    try:
        laptop_cam.disconnect()
        wrist_cam.disconnect()
    except:
        pass
    pygame.quit()
    print("✅ Done.")

if __name__ == "__main__":
    # Suppress pkg_resources warning if possible
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    main()

