import time
import torch
import numpy as np
import cv2
from pathlib import Path
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# --- CONFIGURATION ---
# Your specific checkpoint path
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-01-29/21-33-25_so100_train_50ep/checkpoints/050000/pretrained_model")
DEVICE = "cuda"
FPS = 30

def main():
    print("🚀 Starting Policy Debugger (BLIND MODE)...")
    
    # 1. Load Policy
    print(f"Loading Brain from: {CHECKPOINT_PATH}")
    try:
        policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
        policy.to(DEVICE)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT_PATH)
        print("✅ Brain Loaded.")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return

    # 2. Connect Cameras
    print("Connecting Eyes...")
    try:
        # IDs 0 and 2 are standard for Wrist/Laptop setup
        laptop_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=2, fps=FPS, width=640, height=480, fourcc="MJPG"))
        wrist_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=FPS, width=640, height=480, fourcc="MJPG"))
        laptop_cam.connect()
        wrist_cam.connect()
        print("✅ Eyes Open.")
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    print("\n🔍 STARTING BLIND TURING TEST")
    print("----------------------------------")
    print("1. Wave the red wire in front of the cameras.")
    print("2. Watch the 'ACTION' numbers below.")
    print("   - FROZEN numbers = Normalization/Input Error.")
    print("   - CHANGING numbers = Brain is working.")
    print("----------------------------------")
    print("Press 'Ctrl+C' to quit.\n")

    try:
        while True:
            loop_start = time.perf_counter()
            
            # Read Frames
            laptop_img = laptop_cam.read()
            wrist_img = wrist_cam.read()
            
            if laptop_img is None or wrist_img is None:
                continue
            
            # Prepare Inputs
            laptop_tensor = torch.from_numpy(laptop_img).permute(2, 0, 1).float() / 255.0
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0
            
            # Dummy state (assuming 0 since we aren't moving)
            state_tensor = torch.zeros(6).float()

            observation = {
                "observation.images.laptop": laptop_tensor.unsqueeze(0).to(DEVICE),
                "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
            }
            
            # Ask the Brain
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            
            # Get the "Next Move"
            next_move = action.squeeze(0).cpu().numpy()[:6]
            
            # VISUALIZATION
            # Print numbers to terminal (updates in place)
            print(f"\r🎯 Targets: {np.round(next_move, 2)}   ", end="")
            
            # --- VIDEO WINDOW DISABLED TO PREVENT CRASH ---
            # combined_img = np.hstack((laptop_img, wrist_img))
            # cv2.imshow("Robot Eyes", combined_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            # ----------------------------------------------
                
            # Rate Limit
            dt = time.perf_counter() - loop_start
            time.sleep(max(0, (1/FPS) - dt))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        laptop_cam.disconnect()
        wrist_cam.disconnect()
        # cv2.destroyAllWindows() # Disabled
        print("\nDone.")

if __name__ == "__main__":
    main()
