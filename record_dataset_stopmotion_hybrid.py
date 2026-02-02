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
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    print("🔌 Connecting to Cameras...")
    laptop_config = OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480, fourcc="MJPG")
    wrist_config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480, fourcc="MJPG")
    
    laptop_cam = OpenCVCamera(laptop_config)
    wrist_cam = OpenCVCamera(wrist_config)
    
    try:
        laptop_cam.connect()
        wrist_cam.connect()
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
    WINDOW_W, WINDOW_H = 1280, 480
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("SO-100 Two-Phase Recorder")

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
    print("\n🚀 READY! Two-Phase Recording:")
    print("  [Start]: Start/Stop Recording Episode")
    print("  [X]:     Switch to STOP-MOTION Mode (Permanent)")
    print("  [Y]:     Capture Single Frame (in Stop-Motion Mode)")
    print("  [Back]:  Exit Script")
    print("  [RT]:    Hold for Manual/Limp Mode")
    print("  Sticks:  Move Robot\n")

    current_ep_dir = None
    frame_idx = 0
    recording = False
    
    # Modes: "CONTINUOUS" or "STOP_MOTION"
    rec_mode = "CONTINUOUS"
    
    # Button States
    capture_pressed_prev = False
    switch_pressed_prev = False
    
    running = True
    while running:
        t0 = time.perf_counter()
        
        # --- PYGAME EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.JOYBUTTONDOWN:
                # Back (6) -> Exit
                if event.button == 6:
                    print("🛑 Exit requested.")
                    running = False
                
                # Start (7) -> Toggle Recording Episode
                if event.button == 7:
                    recording = not recording
                    if recording:
                        if current_ep_dir is None:
                            current_ep_dir = ensure_episode_dir(DATASET_ROOT)
                            frame_idx = 0
                            print(f"🔴 EPISODE STARTED: {current_ep_dir}")
                        else:
                            print("🔴 RESUMED RECORDING")
                    else:
                        print("⚪ PAUSED RECORDING")

        # --- READ INPUTS ---
        # 1. Switch Mode (X Button - 2)
        btn_x = joystick.get_button(2)
        if btn_x and not switch_pressed_prev:
            if rec_mode == "CONTINUOUS":
                rec_mode = "STOP_MOTION"
                print("\n>>> SWITCHING TO STOP-MOTION MODE <<<")
                print(">>> Automatic recording STOPPED. Press 'Y' to capture frames.\n")
        switch_pressed_prev = btn_x

        # 2. Hybrid Control Logic (RT Clutch)
        rt_val = joystick.get_axis(5)
        # Default Joystick Values
        ax0 = joystick.get_axis(0)
        ax1 = joystick.get_axis(1)
        ax3 = joystick.get_axis(3)
        ax4 = joystick.get_axis(4)
        lb = joystick.get_button(4) 
        rb = joystick.get_button(5)
        btn_a = joystick.get_button(0)
        btn_b = joystick.get_button(1)
        
        # Limp/Manual Mode
        if rt_val > 0.5:
            fresh_obs = robot.get_observation()
            for name in motor_names:
                key = f"{name}.pos"
                val = fresh_obs.get(key)
                if val is not None:
                     if hasattr(val, "item"): val = val.item()
                     target_joints[name] = val
            # Allow Gripper override
            if btn_a: target_joints["gripper"] = -45
            if btn_b: target_joints["gripper"] = 90
            
        else:
            # Joystick Teleop
            speed = 0.1 # Precision Speed
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

        # --- SENSORS & VIZ ---
        with laptop_cam.frame_lock: laptop_img = laptop_cam.latest_frame
        with wrist_cam.frame_lock: wrist_img = wrist_cam.latest_frame
        robot_obs = robot.get_observation() 

        if laptop_img is not None and wrist_img is not None:
            viz_img = np.hstack((laptop_img, wrist_img))
            viz_surf = pygame.surfarray.make_surface(viz_img.swapaxes(0, 1))
            screen.blit(viz_surf, (0, 0))
            
            # Overlay Info
            font = pygame.font.SysFont(None, 36)
            
            # Mode Indicator
            mode_color = (0, 255, 0) if rec_mode == "CONTINUOUS" else (255, 200, 0)
            mode_text = f"MODE: {rec_mode}"
            screen.blit(font.render(mode_text, True, mode_color), (50, 20))
            
            # Rec Status
            status_text = f"REC: {'ON' if recording else 'OFF'} | Frame: {frame_idx}"
            screen.blit(font.render(status_text, True, (255, 0, 0) if recording else (200, 200, 200)), (50, 50))

            # Motor Positions Table
            font_small = pygame.font.SysFont(None, 24)
            bg_rect = pygame.Surface((220, 180))
            bg_rect.set_alpha(128)
            bg_rect.fill((0, 0, 0))
            screen.blit(bg_rect, (10, 100))
            
            y_offset = 110
            for name in motor_names:
                val = target_joints.get(name, 0.0)
                txt = font_small.render(f"{name}: {val:.1f}", True, (200, 255, 200))
                screen.blit(txt, (20, y_offset))
                y_offset += 25
            
            pygame.display.flip()

            # --- RECORDING LOGIC ---
            should_save = False
            
            if recording and current_ep_dir:
                if rec_mode == "CONTINUOUS":
                    should_save = True
                
                elif rec_mode == "STOP_MOTION":
                    # Check Y Button (3)
                    btn_y = joystick.get_button(3)
                    if btn_y and not capture_pressed_prev:
                        should_save = True
                        print(f"📸 Captured Frame {frame_idx} (Stop-Motion)")
                    capture_pressed_prev = btn_y
            
            if should_save:
                # Save
                laptop_bgr = cv2.cvtColor(laptop_img, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(current_ep_dir / f"laptop_{frame_idx:06d}.jpg"), laptop_bgr)
                cv2.imwrite(str(current_ep_dir / f"wrist_{frame_idx:06d}.jpg"), wrist_bgr)
                
                serializable_obs = {}
                for k, v in robot_obs.items():
                    if hasattr(v, "item"): serializable_obs[k] = v.item()
                    elif isinstance(v, (np.ndarray, list)): serializable_obs[k] = v.tolist() if isinstance(v, np.ndarray) else v
                    else: serializable_obs[k] = v
                
                data_point = {
                    "frame_index": frame_idx,
                    "timestamp": time.time(),
                    "observation": serializable_obs,
                    "action": target_joints
                }
                with open(current_ep_dir / f"frame_{frame_idx:06d}.json", "w") as f:
                    json.dump(data_point, f)
                frame_idx += 1

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    # Cleanup
    print("🧹 Closing connections...")
    try: robot.disconnect()
    except: pass
    try: 
        laptop_cam.disconnect()
        wrist_cam.disconnect()
    except: pass
    pygame.quit()
    print("✅ Done.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    main()
