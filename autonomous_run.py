import time
import torch
import numpy as np
import cv2
import pygame
import argparse
from pathlib import Path
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# Configuration
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-02-02/18-26-42_so100_diffusion_dropout_21ep/checkpoints/020000/pretrained_model")
DEVICE = "cuda"
FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def get_arguments():
    parser = argparse.ArgumentParser(description="Autonomous Run Script for SO-100 Robot")
    parser.add_argument("--viz", type=int, default=1, help="Visualization On/Off (1/0). Default 1.")
    parser.add_argument("--freq", type=int, default=2, help="Query Frequency (Steps between replanning). Default 2.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed scalar (Safety dampener). Default 1.0.")
    parser.add_argument("--x_offset", type=float, default=0.0, help="X (Pan) Offset. Default 0.0.")
    parser.add_argument("--y_offset", type=float, default=0.0, help="Y (Lift) Offset. Default 0.0.")
    parser.add_argument("--z_offset", type=float, default=0.0, help="Z (Elbow) Offset. Default 0.0.")
    parser.add_argument("--policy", type=str, choices=["act", "diffusion"], default="act", help="Policy Type (act/diffusion). Default act.")
    return parser.parse_args()

def main():
    args = get_arguments()
    QUERY_FREQUENCY = args.freq
    SHOW_VISUALIZATION = bool(args.viz)
    SPEED_SCALAR = args.speed
    OFFSETS = [args.x_offset, args.y_offset, args.z_offset]
    POLICY_TYPE = args.policy

    print(f"🚀 Starting Autonomous Run Script...")
    print(f"   [Config] Viz: {SHOW_VISUALIZATION}, Freq: {QUERY_FREQUENCY}, Speed: {SPEED_SCALAR}")
    print(f"   [Offsets] X(Pan): {OFFSETS[0]}, Y(Lift): {OFFSETS[1]}, Z(Elbow): {OFFSETS[2]}")
    print(f"   [Policy] Type: {POLICY_TYPE}")
    
    # 1. Load Policy
    print(f"Loading policy from: {CHECKPOINT_PATH}")
    if POLICY_TYPE == "diffusion":
        policy = DiffusionPolicy.from_pretrained(CHECKPOINT_PATH)
    else:
        policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    
    policy.to(DEVICE)
    policy.eval()
    
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT_PATH)
    print("✅ Policy and processors loaded.")

    # 2. Connect Hardware
    print("Connecting to robot...")
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    
    print("Connecting to cameras...")
    laptop_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=2, fps=FPS, width=640, height=480, fourcc="MJPG"))
    wrist_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=FPS, width=640, height=480, fourcc="MJPG"))
    
    laptop_cam.connect()
    wrist_cam.connect()
    print("✅ Hardware connected.")

    # 3. Setup Telemetry UI (Only if Viz is On)
    screen = None
    font = None
    if SHOW_VISUALIZATION:
        pygame.init()
        screen_w, screen_h = 640, 480
        screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(f"Policy Internals | Freq: {QUERY_FREQUENCY}")
        font = pygame.font.SysFont("monospace", 18)

    # 4. Inference Loop
    print(f"🛑 Press Ctrl+C to stop. Sliding Window: {QUERY_FREQUENCY} steps.")
    
    inference_count = 0
    
    try:
        dt = 1.0 / FPS
        
        while True:
            # --- PHASE A: INFERENCE (Once per QUERY_FREQUENCY) ---
            
            # 1. Read Current Observations
            laptop_img = laptop_cam.read()
            wrist_img = wrist_cam.read()
            
            if laptop_img is None or wrist_img is None:
                print("Warning: Failed to read camera frame.")
                continue

            robot_obs = robot.get_observation()
            state_vec = []
            for name in MOTOR_NAMES:
                val = robot_obs.get(f"{name}.pos")
                if val is None: val = 0.0
                elif hasattr(val, "item"): val = val.item()
                state_vec.append(val)
            
            # 2. Prepare Tensors
            laptop_tensor = torch.from_numpy(laptop_img).permute(2, 0, 1).float() / 255.0
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.tensor(state_vec).float()

            observation = {
                "observation.images.laptop": laptop_tensor.unsqueeze(0).to(DEVICE),
                "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
            }
            
            # 3. Predict Action Chunk
            observation = preprocessor(observation)
            
            # Monitoring Inference
            t_infer_start = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(observation)
            t_infer_end = time.perf_counter()
            
            infer_dur = t_infer_end - t_infer_start
            inference_count += 1
            if inference_count % 100 == 0:
                print(f"Inference: {infer_dur:.3f}s")
            
            if infer_dur > 0.033:
                print(f"⚠️  WARNING: GPU TOO SLOW (>{infer_dur:.3f}s)")
            
            action = postprocessor(action)
            action_np = action.squeeze(0).cpu().numpy()
            
            # 4. Slice Chunk: Take only first N steps
            if action_np.ndim == 1: action_np = action_np[np.newaxis, :]
            steps_to_run = action_np[:QUERY_FREQUENCY]
            
            # --- PHASE B: EXECUTION LOOP (Run N steps) ---
            for i, current_action in enumerate(steps_to_run):
                start_step = time.perf_counter()
                
                # 1. Send Action (With Speed Scalar)
                action_dict = {}
                for j, name in enumerate(MOTOR_NAMES):
                    target_val = current_action[j]
                    
                    # Apply Manual Offsets (to indices 0, 1, 2)
                    if j < 3:
                        target_val += OFFSETS[j]
                    
                    # Apply Speed Scalar (Damping)
                    # New Target = Current + (Target - Current) * Speed
                    # We use state_vec which is the robot position at start of inference.
                    # Ideally we should read fresh obs, but for perf we use state_vec.
                    current_val_est = state_vec[j] 
                    damped_val = current_val_est + (target_val - current_val_est) * SPEED_SCALAR
                    
                    action_dict[f"{name}.pos"] = torch.tensor([damped_val], dtype=torch.float32)
                    
                    # Update state_vec estimate for next step smoothing (simple assumption robot moved)
                    state_vec[j] = damped_val 

                robot.send_action(action_dict)
                
                # 2. Update Visualization (Only if Enabled)
                if SHOW_VISUALIZATION:
                    if i > 0:
                        laptop_img = laptop_cam.read()
                        wrist_img = wrist_cam.read()
                    
                    # Viz: Robot Eyes
                    if laptop_img is not None and wrist_img is not None:
                        vis_laptop = cv2.cvtColor(laptop_img, cv2.COLOR_RGB2BGR)
                        vis_wrist = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                        
                        x1 = (640 - 480) // 2
                        cv2.rectangle(vis_laptop, (x1, 0), (x1+480, 480), (0, 255, 0), 2)
                        cv2.rectangle(vis_wrist, (x1, 0), (x1+480, 480), (0, 255, 0), 2)
                        
                        cv2.imshow("Robot Eyes", np.hstack((vis_laptop, vis_wrist)))
                        cv2.waitKey(1)
                    
                    # Viz: Telemetry
                    screen.fill((30, 30, 30))
                    screen.blit(font.render(f"Step: {i+1}/{QUERY_FREQUENCY} | Inf: {infer_dur*1000:.1f}ms", True, (255, 255, 255)), (10, 10))
                
                    # Bars
                    start_y = 50
                    for k, name in enumerate(MOTOR_NAMES):
                        real_val = state_vec[k] 
                        target_val = current_action[k]
                        y = start_y + k * 50
                        
                        pygame.draw.rect(screen, (60, 60, 60), (0, y, screen_w, 40)) # BG
                        cx = int((0 + 180) / 360 * screen_w)
                        pygame.draw.line(screen, (100, 100, 100), (cx, y), (cx, y+40), 1) # Zero
                        
                        tx = int((target_val + 180) / 360 * screen_w)
                        pygame.draw.rect(screen, (255, 50, 50), (tx-1, y, 2, 40)) # Target
                        
                        screen.blit(font.render(f"{name}: {target_val:.1f}", True, (200, 200, 200)), (10, y + 10))
                        
                    pygame.display.flip()
                    
                    # Pump Events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: raise KeyboardInterrupt
                
                step_dur = time.perf_counter() - start_step
                time.sleep(max(0, dt - step_dur))

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Disconnecting...")
        try:
            robot.disconnect()
            laptop_cam.disconnect()
            wrist_cam.disconnect()
            if SHOW_VISUALIZATION:
                cv2.destroyAllWindows()
                pygame.quit()
        except: pass
        print("Done.")

if __name__ == "__main__":
    main()
