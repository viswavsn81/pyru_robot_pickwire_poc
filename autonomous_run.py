import time
import torch
import numpy as np
import cv2
from pathlib import Path
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Configuration
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-01-30/12-44-18_so100_train_50ep/checkpoints/050000/pretrained_model")
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

    # 3. Inference Loop
    print(f"🛑 Press Ctrl+C to stop. Running at {FPS} FPS...")
    
    try:
        prev_action = None
        while True:
            start_time = time.perf_counter()
            
            # Read Sensors
            # Rely on blocking read for simplicity, assuming cameras run near 30fps.
            # Ideally use async_read logic if stuttering occurs.
            laptop_img = laptop_cam.read()
            wrist_img = wrist_cam.read()
            
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
            
            # Action Selection: Handle 1D (shape [6]) vs 2D (shape [Time, 6])
            if action_np.ndim == 1:
                current_action = action_np
            else:
                current_action = action_np[0]
            
            # Smoothing (EMA)
            if prev_action is None:
                prev_action = current_action
            
            smoothed_action = (EMA_ALPHA * current_action) + ((1 - EMA_ALPHA) * prev_action)
            prev_action = smoothed_action

            action_dict = {}
            for i, name in enumerate(MOTOR_NAMES):
                # Use the smoothed action value for the specific joint
                val = smoothed_action[i]
                action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)
            
            robot.send_action(action_dict)
            
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
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
