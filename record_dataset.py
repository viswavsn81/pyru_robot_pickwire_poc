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
        # --- READ CONTROLS (Teleoperation) ---
        # Helper for exponential sensitivity (cubic curve)
        def curve(val):
            return val * abs(val) * abs(val)

        # Read Raw Inputs
        ax0 = curve(joystick.get_axis(0)) # LS Left/Right -> Pan
        ax1 = curve(joystick.get_axis(1)) # LS Up/Down    -> Lift OR Elbow (if Z-Lock)
        ax3 = curve(joystick.get_axis(3)) # RS Left/Right -> Wrist Flex
        ax4 = curve(joystick.get_axis(4)) # RS Up/Down    -> Elbow Flex (normal)
        
        lb = joystick.get_button(4) # Wrist Roll Left
        rb = joystick.get_button(5) # Z-Lock Modifier / Wrist Roll Right
        
        btn_a = joystick.get_button(0) # Close Gripper
        btn_b = joystick.get_button(1) # Open Gripper

        # Base Speed
        speed = 1.0  # Increased base speed because cubic curve dampens small inputs

        # 1. Shoulder Pan (Base Rotation)
        if abs(ax0) > 0.01: 
            target_joints["shoulder_pan"] += ax0 * speed

        # 2. Z-Plane Lock Logic (Right Bumper Modifier)
        if rb:
            # LOCKED MODE: RB Held -> LS Up/Down moves ELBOW (Reach), maintaining height
            # We disable Shoulder Lift control here to "lock" height (conceptually)
            # Actually user asked to move Elbow Flex to slide flatly.
            if abs(ax1) > 0.01:
                target_joints["elbow_flex"] -= ax1 * speed
            
            # Allow Wrist Roll with LB/RB? 
            # If RB is Shift, we lose it as a button. 
            # Let's map Wrist Roll to Triggers or something else?
            # User instruction: "IF RB is Held... IF RB is Released".
            # Implies RB is *only* a modifier now.
            # So we need a new way to roll right? 
            # Let's use LB for Left, and maybe Triggers for Roll?
            # Or just keep LB for Roll Left, and accept we lost Roll Right on RB?
            # User didn't specify replacement for Roll Right. 
            # I will map simple logic: 
            # LB = Roll Left
            # (No Roll Right button available on Bumpers now)
            pass
        else:
            # NORMAL MODE: RB Released
            # LS Up/Down moves Shoulder Lift (Height)
            if abs(ax1) > 0.01:
                target_joints["shoulder_lift"] -= ax1 * speed
                
            # RS Up/Down moves Elbow Flex
            if abs(ax4) > 0.01:
                target_joints["elbow_flex"] -= ax4 * speed

            # RB as Roll Right (Normal)
            # Wait, if RB is modifier, we can't use it for Roll Right simultaneously without conflict.
            # User said "Use RB as Shift Key". 
            # This implies RB is NO LONGER a functional button for other things.
            # I will disable Roll Right on RB to avoid erratic behavior.
            # Alternative: Use Triggers (Axis 2/5) for Roll?
            # For strict compliance, I follows the Z-lock instruction.
            # BUT I'll add a fallback: 
            # Let's use Trigger axes for Roll if available, or just single-direction roll?
            # I'll enable Roll Right if LB is NOT held, maybe? No.
            # I will map Roll to Triggers (Axis 2 = LT, Axis 5 = RT on Xbox)
            pass

        # 3. Wrist Flex (RS Left/Right)
        if abs(ax3) > 0.01:
            target_joints["wrist_flex"] -= ax3 * speed

        # 4. Wrist Roll (Triggers for reliability)
        trigger_l = joystick.get_axis(2) # LT
        trigger_r = joystick.get_axis(5) # RT
        
        # Win/Linux trigger mapping varies (-1 to 1 or 0 to 1). 
        # Usually -1 is released, 1 is pressed.
        if trigger_l > 0.1: target_joints["wrist_roll"] -= speed * 0.5
        if trigger_r > 0.1: target_joints["wrist_roll"] += speed * 0.5

        # 5. Gripper
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

