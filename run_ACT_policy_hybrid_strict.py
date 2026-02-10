import time
import torch
import numpy as np
import cv2
import pygame
import argparse
from pathlib import Path
from threading import Thread, Lock
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# --- CONFIGURATION ---
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-02-04/15-26-22_act_strict_grasp/checkpoints/020000/pretrained_model")

DEVICE = "cuda"
FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

class ThreadedCamera:
    def __init__(self, src, api_preference=cv2.CAP_ANY, name="Camera"):
        self.name = name
        self.src = src
        print(f"[{name}] Opening Camera: {src} ...")
        self.cap = cv2.VideoCapture(src, api_preference)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
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
        if self.started: return self
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

def perform_safe_startup(robot):
    print("⚠️  Moving to Safe Startup Posture...")
    obs = robot.get_observation()
    action_dict = {}
    for i, name in enumerate(MOTOR_NAMES):
        val = obs.get(f"{name}.pos")
        if hasattr(val, "item"): val = val.item()
        if name == "elbow_flex":
            val += -20.0
        action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)
    robot.send_action(action_dict)
    time.sleep(2.0)
    print("✅ Safe Posture Reached.")

def get_arguments():
    parser = argparse.ArgumentParser(description="ACT Policy Runner with Hybrid Macro")
    parser.add_argument("--offset-pan", type=float, default=0.0, help="Offset for Shoulder Pan")
    parser.add_argument("--offset-lift", type=float, default=0.0, help="Offset for Shoulder Lift")
    parser.add_argument("--offset-elbow", type=float, default=0.0, help="Offset for Elbow Flex")
    parser.add_argument("--slow-speed", type=float, default=0.3, help="Speed scale when S is held")
    return parser.parse_args()

def main():
    args = get_arguments()
    OFFSETS = [args.offset_pan, args.offset_lift, args.offset_elbow]

    print(f"🚀 Starting ACT Policy Runner (Hybrid Strict)...")
    print(f"   [Model] {CHECKPOINT_PATH}")
    print(f"   [Offsets] Pan: {OFFSETS[0]}, Lift: {OFFSETS[1]}, Elbow: {OFFSETS[2]}")
    
    # 1. Load Policy
    print(f"Loading ACT Policy...")
    try:
        policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
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
        robot = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True))
        robot.connect()
        print("✅ Robot Connected!")
        perform_safe_startup(robot)
    except Exception as e:
        print(f"❌ Robot connection failed: {e}")
        return

    print("🔌 Connecting to Cameras...")
    desk_idx = find_desk_camera()
    if desk_idx is None: return
    desk_cam = ThreadedCamera(desk_idx, cv2.CAP_V4L2, name="Desk")
    wrist_url = "udp://10.42.0.1:3000?fifo_size=5000000&overrun_nonfatal=1"
    wrist_cam = ThreadedCamera(wrist_url, cv2.CAP_FFMPEG, name="Wrist")
    
    try:
        desk_cam.start()
        wrist_cam.start()
        time.sleep(1)
    except Exception as e:
        print(f"❌ Camera connection failed: {e}")
        return

    # 3. Setup Pygame
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("ACT Runner (Hybrid) | HOLD SPACE")
    font = pygame.font.SysFont("monospace", 18)

    print("\n🟢 SYSTEM READY.")
    print("   HOLD [SPACEBAR] TO ENABLE ROBOT MOVEMENT.")
    
    # Initialize Smoothed Action Tracker
    obs = robot.get_observation()
    smoothed_action = []
    for name in MOTOR_NAMES:
        val = obs.get(f"{name}.pos", 0.0)
        if hasattr(val, "item"): val = val.item()
        smoothed_action.append(val)
    smoothed_action = np.array(smoothed_action, dtype=np.float32)

    # --- MACRO STATE VARIABLES ---
    macro_state = 0 # 0=AI, 1=Push, 2=Lift, 3=Drop
    macro_timer = 0.0
    macro_base_pos = None

    running = True
    try:
        while running:
            t0 = time.perf_counter()
            
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False
            
            safety_enabled = keys[pygame.K_SPACE]
            slow_mode = keys[pygame.K_s]
            current_speed_scale = args.slow_speed if slow_mode else 1.0

            # --- SENSORS & INFERENCE ---
            desk_img = desk_cam.read()
            wrist_img = wrist_cam.read()
            if desk_img is None or wrist_img is None: continue
            
            desk_img = cv2.resize(desk_img, (640, 480))
            wrist_img = cv2.resize(wrist_img, (640, 480))

            robot_obs = robot.get_observation()
            state_vec = []
            for name in MOTOR_NAMES:
                val = robot_obs.get(f"{name}.pos", 0.0)
                if hasattr(val, "item"): val = val.item()
                state_vec.append(val)

            # Inference
            desk_tensor = torch.from_numpy(cv2.cvtColor(desk_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            wrist_tensor = torch.from_numpy(cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.tensor(state_vec).float()
            
            observation = {
                "observation.images.laptop": desk_tensor.unsqueeze(0).to(DEVICE),
                "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
            }

            with torch.inference_mode():
                observation = preprocessor(observation)
                action = policy.select_action(observation)
                action = postprocessor(action)
            action_np = action.squeeze(0).detach().cpu().numpy()
            if action_np.ndim == 2: target_action = action_np[0] 
            else: target_action = action_np
            
            # --- EXECUTION ---
            if safety_enabled:
                # 1. Prepare AI Target (Target + Offsets)
                full_target = target_action.copy()
                for i in range(3): 
                    full_target[i] += OFFSETS[i]
                
                # --- MACRO LOGIC ---
                macro_color = (255, 100, 0)
                status_msg = "RUNNING (AI)"
                
                # Check Trigger
                if macro_state == 0 and full_target[5] < 75.0: # Close detected
                    print("\n⚡ MACRO TRIGGERED: GRASP INITIATED")
                    macro_state = 1
                    macro_base_pos = smoothed_action.copy() # Lock position
                    macro_timer = time.time()
                
                if macro_state > 0:
                    delta = time.time() - macro_timer
                    
                    # Ensure base pos is valid
                    final_cmd = macro_base_pos.copy()
                    
                    if macro_state == 1: # PUSH (0.0 - 0.5s)
                        status_msg = "MACRO: PUSHING"
                        final_cmd[1] += 5.0 # Push Down (Shoulder Lift +)
                        final_cmd[5] = -45.0 # Close Gripper
                        if delta > 0.5:
                            macro_state = 2
                            
                    elif macro_state == 2: # LIFT (0.5 - 1.5s)
                        status_msg = "MACRO: LIFTING"
                        final_cmd[1] -= 25.0 # Lift Up (Shoulder Lift -)
                        final_cmd[2] -= 25.0 # Lift Elbow
                        final_cmd[5] = -45.0 # Keep Closed
                        if delta > 1.5:
                            macro_state = 3
                            
                    elif macro_state == 3: # DROP (1.5s +)
                        status_msg = "MACRO: DROPPING"
                        final_cmd[1] -= 25.0 
                        final_cmd[2] -= 25.0
                        final_cmd[5] = 90.0 # Open Gripper
                        
                    # Override Target
                    # In macro mode, we bypass EMA for snappier response, OR we can simply
                    # set smoothed_action = final_cmd to force it.
                    # User request: "Ensure smoothed_action is updated to this new target"
                    smoothed_action = final_cmd
                    
                else:
                    # AI Mode: Apply EMA
                    alpha = current_speed_scale
                    smoothed_action = (smoothed_action * (1.0 - alpha)) + (full_target * alpha)
                    if slow_mode: 
                        status_msg = f"SLOW MODE ({current_speed_scale}x)"
                        status_color = (0, 150, 255)
                    else:
                        status_color = (0, 255, 0)
                
                # Send Action
                action_dict = {}
                for i, name in enumerate(MOTOR_NAMES):
                    val = smoothed_action[i]
                    if name != "gripper": val = np.clip(val, -175, 175)
                    action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)

                robot.send_action(action_dict)
                
            else:
                # Reset if Safety Released
                if macro_state > 0:
                    print("MACRO RESET (Safety Released)")
                    macro_state = 0
                
                status_msg = "PAUSED (HOLD SPACE)"
                status_color = (255, 50, 50)

            # --- VIZ ---
            desk_small = cv2.resize(desk_img, (320, 240))
            wrist_small = cv2.resize(wrist_img, (320, 240))
            viz_cam = np.hstack((desk_small, wrist_small))
            viz_cam = cv2.cvtColor(viz_cam, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(viz_cam.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            
            # Draw Status Bar
            if macro_state > 0: status_color = (255, 140, 0) # Orange for Macro
            
            msg_surf = font.render(status_msg, True, status_color)
            pygame.draw.rect(screen, (0,0,0), (0, 240, 640, 40))
            screen.blit(msg_surf, (20, 250))
            pygame.display.flip()

            dt = time.perf_counter() - t0
            time.sleep(max(0, 1.0/FPS - dt))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: robot.disconnect()
        except: pass
        try: desk_cam.disconnect(); wrist_cam.disconnect()
        except: pass
        pygame.quit()
        print("✅ Done.")

if __name__ == "__main__":
    main()
