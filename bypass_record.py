import sys
import os
import time
import cv2
import numpy as np
import logging
from pathlib import Path  # <--- THE MISSING PIECE

# 1. FORCE PYTHON TO LOOK IN 'SRC'
sys.path.append(os.path.abspath("src"))

# 2. EXACT IMPORTS
print("🔍 Importing modules...")
try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    print("✅ All modules found!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# 3. SETUP LOGGING
logging.basicConfig(level=logging.INFO)
print("🚀 STARTING BYPASS RECORDER...")

def record_smoke_test():
    # 4. CONNECT CAMERAS
    print("📷 Connecting Cameras...")
    try:
        cam_laptop = OpenCVCamera(OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480))
        cam_wrist = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480))
        cam_laptop.connect()
        cam_wrist.connect()
        print("✅ Cameras Connected")
    except Exception as e:
        print(f"❌ Camera Error: {e}")

    # 5. CONNECT ROBOT
    print("🔌 Connecting Robot...")
    try:
        # THE FIX: Wrap the path string in Path()
        calib_path = Path(".cache/huggingface/lerobot/calibration/robots/so_follower")
        
        robot_config = SOFollowerRobotConfig(
            port="/dev/ttyACM0",
            calibration_dir=calib_path  # <--- Passed as a Path object, not string
        )
        
        robot = SOFollower(robot_config)
        robot.connect()
        print("✅ Robot Connected")
    except Exception as e:
        print(f"❌ Robot Connection Error: {e}")
        return

    # 6. INJECT MOTOR MODELS
    print("🔧 Injecting Motor Models...")
    try:
        robot.bus.set_model(1, "xm430-w350")
        robot.bus.set_model(2, "xm430-w350")
        robot.bus.set_model(3, "xm430-w350")
        robot.bus.set_model(4, "xl330-m288")
        robot.bus.set_model(5, "xl330-m288")
        robot.bus.set_model(6, "xl330-m077")
        print("✅ Models Injected")
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    # 7. MAIN LOOP
    print("\n🎥 === LIVE FEED ===")
    print("   Press 'SPACE' to check robot status.")
    print("   Press 'Q' to Quit.")
    
    while True:
        img_l = cam_laptop.read()
        img_w = cam_wrist.read()
        
        if img_l is not None and img_w is not None:
            vis = np.hstack((img_l, img_w))
            cv2.imshow("Bypass Recorder", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            try:
                state = robot.read()
                # Check for 'dof_pos' or 'motor_positions'
                if hasattr(state, 'dof_pos'):
                    val = state.dof_pos[5]
                elif hasattr(state, 'motor_positions'):
                    val = state.motor_positions[5]
                else:
                    val = "Unknown Format"
                print(f"✅ Robot Alive! Gripper: {val}")
            except Exception as e:
                print(f"❌ Read Error: {e}")

    print("🛑 Disconnecting...")
    robot.disconnect()
    cam_laptop.disconnect()
    cam_wrist.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_smoke_test()
