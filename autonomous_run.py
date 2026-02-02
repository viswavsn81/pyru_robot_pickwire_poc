import time
import torch
import numpy as np
import cv2
import pygame
from pathlib import Path
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Configuration
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-02-01/20-59-27_so100_dropout_14ep/checkpoints/010000/pretrained_model")
DEVICE = "cuda"
FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EMA_ALPHA = 0.5

def main():
    print("🚀 Starting Autonomous Run Script...")
    
    # 1. Load Policy
    print(f"Loading policy from: {CHECKPOINT_PATH}")
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()
    
    # Load processors
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT_PATH)
    print("✅ Policy and processors loaded.")

    # 2. Connect Hardware
    print("Connecting to robot...")
    # Using SO100Follower as requested (matching smoke_test_v2.py)
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot connected.")

    print("Connecting to cameras...")
    # Laptop: 2, Wrist: 0 (Manual matches smoke_test_v2.py)
    # Using MJPEG for performance matching record_dataset.py
    laptop_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=2, fps=FPS, width=640, height=480, fourcc="MJPG"))
    wrist_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=FPS, width=640, height=480, fourcc="MJPG"))
    
    laptop_cam.connect()
    wrist_cam.connect()
    print("✅ Cameras connected.")

    # 4. Setup Telemetry UI
    pygame.init()
    screen_w, screen_h = 640, 480
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Policy Internals (Blue=Real, Red=Target)")
    font = pygame.font.SysFont("monospace", 18)

    # 3. Inference Loop
    print(f"🛑 Press Ctrl+C to stop. Running at {FPS} FPS...")
    
    try:
        prev_action = None
        action_history = [] # Buffer for Temporal Ensembling
        dt = 1.0 / FPS
        while True:
            start_time = time.perf_counter()
            
            # Read Sensors
            # Rely on blocking read for simplicity, assuming cameras run near 30fps.
            # Ideally use async_read logic if stuttering occurs.
            laptop_img = laptop_cam.read()
            wrist_img = wrist_cam.read()
            
            # --- LIVE ROBOT EYES (DEBUG) ---
            if laptop_img is not None and wrist_img is not None:
                # 1. Color Correction (RGB -> BGR for OpenCV Display)
                # The camera class returns RGB, but cv2.imshow expects BGR.
                vis_laptop = cv2.cvtColor(laptop_img, cv2.COLOR_RGB2BGR)
                vis_wrist = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                
                # 2. Draw "Active Area" (Center Crop)
                # Input: 640x480, Crop: 480x480 (Center)
                x1 = (640 - 480) // 2
                x2 = x1 + 480
                cv2.rectangle(vis_laptop, (x1, 0), (x2, 480), (0, 255, 0), 2)
                cv2.rectangle(vis_wrist, (x1, 0), (x2, 480), (0, 255, 0), 2)
                
                # 3. Stack & Display
                combined_img = np.hstack((vis_laptop, vis_wrist))
                cv2.imshow("Robot Eyes", combined_img)
                cv2.waitKey(1)
            # -------------------------------
            
            if laptop_img is None or wrist_img is None:
                print("Warning: Failed to read camera frame.")
                continue

            robot_obs = robot.get_observation()
            
            # Construct Observation State Vector
            state_vec = []
            for name in MOTOR_NAMES:
                # get_observation keys are usually 'joint_name.pos'
                val = robot_obs.get(f"{name}.pos")
                if val is None:
                    # In case key is missing (e.g. simulation vs real diffs), but SO100Follower should provide it.
                    print(f"Warning: Missing observation key {name}.pos")
                    val = 0.0
                elif hasattr(val, "item"):
                    val = val.item()
                state_vec.append(val)
            
            # Convert to Tensors (B, C, H, W) and (B, D)
            # Normalize images to [0,1]
            laptop_tensor = torch.from_numpy(laptop_img).permute(2, 0, 1).float() / 255.0
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.tensor(state_vec).float()

            observation = {
                "observation.images.laptop": laptop_tensor.unsqueeze(0).to(DEVICE),
                "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
            }
            
            # Preprocess
            observation = preprocessor(observation)
            
            # Inference
            with torch.inference_mode():
                action = policy.select_action(observation)
            
            # Postprocess
            action = postprocessor(action)
            
            # Map Output to Robot
            # Map Output to Robot
            # action shape is (Batch, Time, Dims) -> (Time, Dims)
            action_np = action.squeeze(0).cpu().numpy()
            
            # Action Selection: Temporal Ensembling
            # Handle 1D (shape [6]) vs 2D (shape [Time, 6])
            if action_np.ndim == 1:
                action_np = action_np[np.newaxis, :]
            
            # Add current prediction to history
            action_history.append(action_np)
            
            # Average overlapping actions
            # We iterate backwards: index 0 = current frame prediction
            # index i = prediction made i frames ago.
            # For a chunk made 'i' frames ago, we need its action at index 'i'.
            ensemble_action = np.zeros_like(action_np[0])
            count = 0
            
            # Prune and Aggregate
            new_history = []
            for i, pred_chunk in enumerate(reversed(action_history)):
                if i < len(pred_chunk):
                    # This chunk is still valid (has an action for the current relative time)
                    ensemble_action += pred_chunk[i]
                    count += 1
                    new_history.append(pred_chunk) # Keep it (temporarily reversed order)
            
            # Restore correct order for next loop (reverse of reversed = original order)
            action_history = new_history[::-1]
            
            if count > 0:
                smoothed_action = ensemble_action / count
            else:
                smoothed_action = action_np[0] # Fallback
            
            # prev_action = smoothed_action # (Not strictly needed for Ensembling, but harmless)

            action_dict = {}
            for i, name in enumerate(MOTOR_NAMES):
                # Use the smoothed action value for the specific joint
                val = smoothed_action[i]
                action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)
            
            robot.send_action(action_dict)
            
            # --- VISUALIZATION (Telemetry HUD) ---
            screen.fill((30, 30, 30)) # Dark Grey Background
            
            # Draw header
            text = font.render(f"Step Time: {dt*1000:.1f}ms | FPS: {1/dt:.1f}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            
            # Draw Bars for each motor
            bar_h = 40
            margin = 10
            start_y = 50
            
            for i, name in enumerate(MOTOR_NAMES):
                # Data
                real_val = state_vec[i]
                target_val = smoothed_action[i]
                
                # Normalize to screen width (Range: -180 to 180)
                # Center (0) at screen_w / 2
                def val_to_x(v):
                    # Map -180..180 to 0..screen_w
                    return int((v + 180) / 360 * screen_w)
                
                y = start_y + i * (bar_h + margin)
                
                # 1. Background Bar (Range)
                pygame.draw.rect(screen, (60, 60, 60), (0, y, screen_w, bar_h))
                
                # 2. Zero Line (Center)
                center_x = val_to_x(0)
                pygame.draw.line(screen, (100, 100, 100), (center_x, y), (center_x, y+bar_h), 1)
                
                # 3. Real Value (Blue Line)
                real_x = val_to_x(real_val)
                pygame.draw.rect(screen, (50, 100, 255), (real_x-2, y, 4, bar_h))
                
                # 4. Target Value (Red Line)
                target_x = val_to_x(target_val)
                pygame.draw.rect(screen, (255, 50, 50), (target_x-1, y, 2, bar_h))
                
                # Label
                label = font.render(f"{name}: {real_val:.1f} -> {target_val:.1f}", True, (200, 200, 200))
                screen.blit(label, (10, y + 10))

            pygame.display.flip()
            
            # Pump events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # Rate limiting
            dt = time.perf_counter() - start_time
            sleep_time = max(0, (1/FPS) - dt)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Disconnecting...")
        try:
            robot.disconnect()
            laptop_cam.disconnect()
            wrist_cam.disconnect()
            cv2.destroyAllWindows()
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
