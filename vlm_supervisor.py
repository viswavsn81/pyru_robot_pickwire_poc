import time
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# --- Configuration ---
# TODO: User will fill in API key
genai.configure(api_key="AIzaSyCQvjTEg8ksNuejD-OGPL8yFXqQFFqt70U")

# FIXED: Changed model name to Lite version for better rate limits
VLM_MODEL_NAME = "gemini-2.0-flash-lite"
# Update this path if it changed
CHECKPOINT_PATH = Path("/home/pyru/lerobot/outputs/train/2026-01-29/07-33-28_so100_train_40ep/checkpoints/040000/pretrained_model")
DEVICE = "cuda"
FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# --- DEBUG SETTINGS ---
# Set to True if the wire looks BLUE in 'supervisor_view.jpg'
FLIP_COLORS = False  

def ask_supervisor(image_np):
    """
    Sends the image to Gemini.
    Saves 'supervisor_view.jpg' locally for debugging.
    """
    try:
        # 1. HANDLE COLOR CHANNELS
        if FLIP_COLORS:
            # If the user says colors are swapped, flip them before processing
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 2. SAVE DEBUG IMAGE
        # Convert to BGR for OpenCV saving so the user sees the true colors
        debug_save = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("supervisor_view.jpg", debug_save)

        # 3. PREPARE FOR AI
        image_pil = Image.fromarray(image_np)
        
        # 4. RELAXED PROMPT + REASONING
        model = genai.GenerativeModel(VLM_MODEL_NAME)
        response = model.generate_content([
            "Look at this image. I am looking for a RED WIRE on a BLUE CIRCUIT BOARD. Is this specific setup visible and reachable? First explain what you see, then answer YES or NO.",
            image_pil
        ])
        
        text = response.text.strip().upper()
        print(f"🤖 Supervisor Reasoning: {response.text.strip()}")
        
        if "YES" in text:
            return True
        return False
    except Exception as e:
        if "429" in str(e):
            print("⏳ Rate Limit Hit. Cooling down for 30 seconds...")
            time.sleep(30)
        else:
            print(f"❌ Supervisor Error: {e}")
        return False

def main():
    print("🚀 Starting VLM-Supervised Autonomous Script (Debug Mode)...")
    
    # 1. Load Policy
    print(f"Loading policy from: {CHECKPOINT_PATH}")
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT_PATH)
    print("✅ Policy loaded.")

    # 2. Connect Hardware
    print("Connecting to robot...")
    robot = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", id="my_arm", use_degrees=True))
    robot.connect()
    print("✅ Robot connected.")

    print("Connecting to cameras...")
    laptop_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=2, fps=FPS, width=640, height=480, fourcc="MJPG"))
    wrist_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=FPS, width=640, height=480, fourcc="MJPG"))
    laptop_cam.connect()
    wrist_cam.connect()
    
    time.sleep(2) # Warmup
    print("✅ Cameras warm.")

    
    # EMA Smoothing Variable
    prev_action = None

    try:
        while True:
            # --- STATE 1: THINKING ---
            print("\n🤔 Thinking... (Check 'supervisor_view.jpg')")
            
            laptop_img = laptop_cam.read()
            if laptop_img is None: continue
            
            # Ask Supervisor
            safe_to_act = ask_supervisor(laptop_img)
            
            # --- STATE 2: ACTING ---
            if not safe_to_act:
                print("🛑 Supervisor says: WAIT.")
                time.sleep(2.0)
                continue
            
            print("✅ Supervisor says: YES! Acting...")
            
            # Execute 60 steps (2 seconds) then re-check
            for step in range(60):
                loop_start = time.perf_counter()
                
                laptop_img = laptop_cam.read()
                wrist_img = wrist_cam.read()
                if laptop_img is None or wrist_img is None: continue

                robot_obs = robot.get_observation()
                state_vec = []
                for name in MOTOR_NAMES:
                    val = robot_obs.get(f"{name}.pos")
                    if val is None: val = 0.0
                    elif hasattr(val, "item"): val = val.item()
                    state_vec.append(val)
                
                laptop_tensor = torch.from_numpy(laptop_img).permute(2, 0, 1).float() / 255.0
                wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0
                state_tensor = torch.tensor(state_vec).float()

                observation = {
                    "observation.images.laptop": laptop_tensor.unsqueeze(0).to(DEVICE),
                    "observation.images.wrist": wrist_tensor.unsqueeze(0).to(DEVICE),
                    "observation.state": state_tensor.unsqueeze(0).to(DEVICE),
                }
                
                observation = preprocessor(observation)
                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = postprocessor(action)
                
                action = postprocessor(action)
                
                action_np = action.squeeze(0).cpu().numpy()
                
                # --- SMOOTHING (EMA) ---
                if prev_action is None:
                    prev_action = action_np
                
                smoothed_action = (0.7 * action_np) + (0.3 * prev_action)
                prev_action = smoothed_action
                
                action_dict = {}
                for i, name in enumerate(MOTOR_NAMES):
                    # --- SAFETY CLAMPING ---
                    # Clip values between -120 and 120 degrees to prevent damage
                    val = smoothed_action[i]
                    val = np.clip(val, -120, 120)
                    
                    action_dict[f"{name}.pos"] = torch.tensor([val], dtype=torch.float32)
                
                robot.send_action(action_dict)
                
                dt = time.perf_counter() - loop_start
                time.sleep(max(0, (1/FPS) - dt))
            
            print("🏁 Re-evaluating...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()
        laptop_cam.disconnect()
        wrist_cam.disconnect()

if __name__ == "__main__":
    main()
