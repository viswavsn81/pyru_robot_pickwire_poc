import time
import torch
import numpy as np
import cv2
import pygame
import argparse
from pathlib import Path
from threading import Thread, Lock
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# Configuration
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-02-03/13-38-15_diffusion_test_run/checkpoints/001000/pretrained_model")
DEVICE = "cuda"
FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

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
        
        # Buffer size logic for V4L2
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
                    pass
            time.sleep(0.001)

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

def main():
    print(f"🚀 Starting Hybrid Policy Runner...")
    print(f"   [Model] {CHECKPOINT_PATH}")
    
    # 1. Load Policy
    print(f"Loading Diffusion Policy...")
    try:
        policy = DiffusionPolicy.from_pretrained(CHECKPOINT_PATH)
        policy.to(DEVICE)
        policy.eval()
        
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT_PATH)
        print("✅ Policy loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return

    # 2. Connect Hardware
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
    
    # Desk Camera
    desk_idx = find_desk_camera()
    if desk_idx is None: return
    desk_cam = ThreadedCamera(desk_idx, cv2.CAP_V4L2, name="Desk")
    
    # Wrist Camera (FFmpeg)
    # Using confirmed FFmpeg URL
    wrist_url = "udp://10.42.0.1:3000?fifo_size=5000000&overrun_nonfatal=1"
    wrist_cam = ThreadedCamera(wrist_url, cv2.CAP_FFMPEG, name="Wrist")
    
    try:
        desk_cam.start()
        wrist_cam.start()
        time.sleep(1) # Warmup
    except Exception as e:
        print(f"❌ Camera connection failed: {e}")
        return

    # 3. Setup Pygame (Safety & Viz)
    pygame.init()
    screen_w, screen_h = 640, 480
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Hybrid Policy Runner | HOLD SPACE TO MOVE")
    font = pygame.font.SysFont("monospace", 18)

    print("\n🟢 SYSTEM READY.")
    print("   HOLD [SPACEBAR] TO ENABLE ROBOT MOVEMENT.")
    print("   PRESS [Q] TO QUIT.\n")

    # State tracking
    robot_obs = robot.get_observation()
    curr_joints = []
    for name in MOTOR_NAMES:
        val = robot_obs.get(f"{name}.pos", 0.0)
        if hasattr(val, "item"): val = val.item()
        curr_joints.append(val)
        
    running = True
    try:
        while running:
            t0 = time.perf_counter()
            
            # --- EVENTS ---
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            safety_enabled = keys[pygame.K_SPACE]

            # --- SENSORS ---
            desk_img = desk_cam.read()
            wrist_img = wrist_cam.read()
            
            if desk_img is None or wrist_img is None:
                print("⚠️  Frame dropped...")
                continue
            
            # RESIZE IMAGES TO 640x480 (Fix for Tensor Stacking Error)
            desk_img = cv2.resize(desk_img, (640, 480))
            wrist_img = cv2.resize(wrist_img, (640, 480))

            robot_obs = robot.get_observation()
            state_vec = []
            for name in MOTOR_NAMES:
                val = robot_obs.get(f"{name}.pos", 0.0)
                if hasattr(val, "item"): val = val.item()
                state_vec.append(val)

            # --- INFERENCE ---
            # Prepare Batch
            # Convert BGR -> RGB (PyTorch usually expects RGB, OpenCV gives BGR)
            # CHECK: lerobot usually trains on RGB. AND our recorder SAVED images after conversion? 
            # In record_dataset..: `cv2.imwrite` takes BGR. So dataset has BGR JPGs.
            # But `lerobot` video backend reads them. Usually `torchvision.io` reads RGB?
            # CV2 imread reads BGR. 
            # In `lerobot/common/datasets/video_utils.py`, it uses `torchcodec` or `torchvision` or `cv2`.
            # If standard `lerobot` pipeline uses `torchvision` backend, it gets RGB.
            # So if we feed BGR here, colors will be swapped.
            # We should convert to RGB.
            desk_rgb = cv2.cvtColor(desk_img, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
            
            # Normalize [0,1]
            desk_tensor = torch.from_numpy(desk_rgb).permute(2, 0, 1).float() / 255.0
            wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.tensor(state_vec).float()
            
            observation = {
                "observation.images.laptop": desk_tensor.unsqueeze(0).to(DEVICE),
                "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
            }

            # Predict
            with torch.inference_mode():
                # Apply preprocessor usually handled by factory wrapper?
                # In autonomous_run.py: `observation = preprocessor(observation)` was called.
                # Yes, we need to apply it.
                observation = preprocessor(observation)
                action = policy.select_action(observation)
                action = postprocessor(action)
            
            # Action is (1, horizon, dim) or (1, dim)? 
            # Diffusion usually returns (1, horizon, dim)
            action_np = action.squeeze(0).detach().cpu().numpy()
            
            # We just take the first action step? Or sliding window?
            # For simplicity, let's take the first step.
            # Usually we run Receding Horizon Control.
            # Taking index 0 is valid for 30Hz control loop.
            if action_np.ndim == 2: 
                target_action = action_np[0] 
            else:
                target_action = action_np

            # --- EXECUTION ---
            if safety_enabled:
                action_dict = {}
                for i, name in enumerate(MOTOR_NAMES):
                    val = target_action[i]
                    # Clamp for safety
                    if name != "gripper":
                         val = np.clip(val, -175, 175)
                    action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)
                robot.send_action(action_dict)
                status_msg = "RUNNING (SPACE HELD)"
                status_color = (0, 255, 0)
            else:
                status_msg = "PAUSED (HOLD SPACE)"
                status_color = (255, 50, 50)
                # Should we hold current position?
                # Robot stays put by itself if we don't send commands (follower mode inactive? or active holding?)
                # SO-100 `so_follower` writes to `Goal_Position`. If we stop sending, it holds last goal.
                pass

            # --- VIZ ---
            # Show Cameras
            # Only periodically to save perf? Or always?
            # User wants safety, so viz is good.
            # Downscale for split screen
            desk_small = cv2.resize(desk_img, (320, 240))
            wrist_small = cv2.resize(wrist_img, (320, 240))
            viz_cam = np.hstack((desk_small, wrist_small))
            viz_cam = cv2.cvtColor(viz_cam, cv2.COLOR_BGR2RGB)
            
            # Pygame Blit
            surf = pygame.surfarray.make_surface(viz_cam.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            
            # Status Overlay
            msg_surf = font.render(status_msg, True, status_color)
            pygame.draw.rect(screen, (0,0,0), (0, 240, 640, 40))
            screen.blit(msg_surf, (20, 250))
            
            # Action Bars
            start_y = 290
            for k, name in enumerate(MOTOR_NAMES):
                curr = state_vec[k]
                targ = target_action[k]
                y = start_y + k * 25
                
                # BG
                pygame.draw.rect(screen, (50,50,50), (100, y, 400, 20))
                # Center
                cx = 100 + 200
                pygame.draw.line(screen, (100,100,100), (cx, y), (cx, y+20), 1)
                
                # Current (White)
                cv = int(100 + (curr + 180)/360 * 400)
                pygame.draw.rect(screen, (255,255,255), (cv-2, y, 4, 20))
                
                # Target (Red, if running)
                if safety_enabled:
                    tv = int(100 + (targ + 180)/360 * 400)
                    pygame.draw.rect(screen, (255,0,0), (tv-2, y+5, 4, 10))
                
                screen.blit(font.render(name, True, (200,200,200)), (5, y))

            pygame.display.flip()

            # FPS Cap
            dt = time.perf_counter() - t0
            time.sleep(max(0, 1.0/FPS - dt))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("🧹 Cleaning up...")
        try: robot.disconnect()
        except: pass
        try: 
            desk_cam.disconnect()
            wrist_cam.disconnect()
        except: pass
        pygame.quit()
        print("✅ Done.")

if __name__ == "__main__":
    main()
