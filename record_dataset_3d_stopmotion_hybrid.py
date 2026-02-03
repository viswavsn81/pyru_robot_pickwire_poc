#!/usr/bin/env python

import os
import time
import json
import cv2
import torch
import numpy as np
import pygame
from pathlib import Path
from threading import Thread, Lock
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

# Configuration
FPS = 30
DATASET_ROOT = "dataset"

import argparse

class ThreadedCamera:
    """
    Reads frames from a cv2.VideoCapture in a separate thread to prevent blocking.
    """
    def __init__(self, src, api_preference=cv2.CAP_ANY, name="Camera"):
        self.name = name
        self.src = src
        print(f"[{name}] Opening Camera: {src} ...")
        self.cap = cv2.VideoCapture(src, api_preference)
        
        # Try setting common properties (might not work for all backends)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Buffer size logic for GStreamer appsink usually handled in pipeline, 
        # but for V4L2 setting buffer size can help latency.
        if api_preference == cv2.CAP_V4L2:
             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            print(f"[{name}] ❌ Failed to read initial frame!")
        else:
            print(f"[{name}] ✅ Camera started.")

        self.started = False
        self.lock = Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.lock:
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    # If we lose connection, maybe wait a bit
                    pass
            time.sleep(0.001) # Yield slightly

    def read(self):
        with self.lock:
            if self.frame is None: return None
            return self.frame.copy()

    def disconnect(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

def find_desk_camera():
    """Scans common /dev/video indices for a working camera."""
    candidates = [0, 2, 4, 1, 3, 5, 6]
    print(f"🔎 Scanning for Desk Camera (Indices: {candidates})...")
    
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                print(f"✅ Found Desk Camera at Index: {idx}")
                return idx
            cap.release()
    
    print("❌ Could not find any working Desk Camera!")
    return None

def get_arguments():
    parser = argparse.ArgumentParser(description="3D Stop-Motion Hybrid Recorder")
    parser.add_argument("--desk_idx", type=int, default=None, help="Force Desk Camera Index (e.g. 0, 2)")
    parser.add_argument("--wrist_port", type=int, default=3000, help="StereoPi UDP Port (Default 3000)")
    return parser.parse_args()

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
    args = get_arguments()

    # -----------------------------
    # 1. SETUP ROBOT & CAMERAS
    # -----------------------------
    print("🔌 Connecting to Robot...")
    try:
        robot = SO100Follower(SO100FollowerConfig(
            port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
        ))
        robot.connect()
        print("✅ Robot Connected!")
    except Exception as e:
        print(f"❌ Robot connection failed: {e}")
        return

    print("🔌 Connecting to Cameras...")
    
    # 1. Desk Camera
    desk_idx = args.desk_idx
    if desk_idx is None:
        desk_idx = find_desk_camera()
    
    if desk_idx is None:
        print("❌ Cannot proceed without Desk Camera. Check USB connections.")
        return

    desk_cam = ThreadedCamera(desk_idx, cv2.CAP_V4L2, name="Desk")
    
    # 2. Wrist Camera (StereoPi FFmpeg)
    # Using FFmpeg backend as confirmed by diagnose_camera.py
    wrist_url = f"udp://10.42.0.1:{args.wrist_port}?fifo_size=5000000&overrun_nonfatal=1"
    print(f"Connecting to Wrist Camera via FFmpeg: {wrist_url} ...")
    wrist_cam = ThreadedCamera(wrist_url, cv2.CAP_FFMPEG, name="Wrist")
    
    try:
        # Start Threads
        if not desk_cam.grabbed:
             raise Exception("Desk Camera failed to grab initial frame.")
        if not wrist_cam.grabbed:
             print("⚠️  Warning: Wrist Camera (StereoPi) not detected yet. Waiting...")
             # We allow proceeding but warned. Or should we block? User script usually blocks.
             # The ThreadedCamera init already printed failure if grabbed is false.
             # If it failed, let's suggest checking the stream.
             print(f"👉 Check if StereoPi is streaming to port {args.wrist_port}!")
        
        desk_cam.start()
        wrist_cam.start()
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
    pygame.display.set_caption("SO-100 3D Stop-Motion Hybrid Recorder")

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
                            # Create subfolders for organization
                            (current_ep_dir / "images/chunk_0").mkdir(parents=True, exist_ok=True)
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
            speed = 0.5 # Precision Speed (0.1 was very slow, user might prefer faster)
            # Actually user didn't ask to change speed, sticking to original 0.1 logic if possible? 
            # Original was 0.1 in the script I read. I'll keep 0.1 to be safe, but 0.5 is usually more usable.
            # I will double check the original file content I read.
            # "speed = 0.1 # Precision Speed" -> OK I will stick to 0.1
            speed = 0.1 
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
        desk_bgr = desk_cam.read()
        wrist_bgr = wrist_cam.read()
        robot_obs = robot.get_observation() 
        
        # NOTE: ThreadedCamera returns BGR (from cv2).
        # We need RGB for Pygame display.
        
        if desk_bgr is not None and wrist_bgr is not None:
            # Resize if needed (Just in case cameras are different res)
            if desk_bgr.shape != (480, 640, 3): desk_bgr = cv2.resize(desk_bgr, (640, 480))
            if wrist_bgr.shape != (480, 640, 3): wrist_bgr = cv2.resize(wrist_bgr, (640, 480))

            # Convert to RGB for Visualization
            desk_rgb = cv2.cvtColor(desk_bgr, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            
            viz_img = np.hstack((desk_rgb, wrist_rgb))
            
            # Pygame expects (W, H, 3) and we have (H, W, 3) -> transpose
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
            
            # Add Cam Labels
            screen.blit(font.render("Desk (USB)", True, (255,255,255)), (20, 440))
            screen.blit(font.render("Wrist (StereoPi)", True, (255,255,255)), (660, 440))

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
                # Save Images (BGR is standard for cv2.imwrite, so we use the originals)
                # Filename naming convention: laptop_ / wrist_ 
                # User asked for: "inside images/desk and images/wrist" or simple structure?
                # Prompt: "e.g., inside images/desk and images/wrist"
                # The original script flattened them: laptop_0000.jpg
                # I will stick to original flattened structure for compatibility unless explicitly forced.
                # Actually, "e.g. inside images/desk" suggests they might want folders.
                # But to preserve "Keep Existing Logic", I should check if the original script did folders.
                # Original script: current_ep_dir / f"laptop_{frame_idx:06d}.jpg"
                # I will stick to that to avoid breaking training scripts that expect this structure.
                # But wait, user said "Update Desk Camera" -> maybe "laptop" name is confusing?
                # I'll use "laptop" -> "desk" mapping if needed, but `lerobot` usually expects `laptop` / `phone`.
                # If I change filenames to `desk_` / `wrist_`, I might break training config.
                # I will name them `laptop_` (for desk) and `wrist_` (for wrist) to match standard LeRobot config.
                
                cv2.imwrite(str(current_ep_dir / f"laptop_{frame_idx:06d}.jpg"), desk_bgr)
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
        desk_cam.disconnect()
        wrist_cam.disconnect()
    except: pass
    pygame.quit()
    print("✅ Done.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    main()
