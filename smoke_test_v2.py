#!/usr/bin/env python

import time
import numpy as np
import cv2
import torch
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

def main():
    print("🔌 Connecting to Robot (Logic from teleoperate_full_control.py)...")
    # EXACT COPY of connection logic from teleoperate_full_control.py
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    ))
    robot.connect()
    print("✅ Robot Connected!")

    # Connect Cameras
    print("🔌 Connecting to Cameras...")
    # Laptop Camera: Index 2
    # Wrist Camera: Index 0
    laptop_config = OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480)
    wrist_config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    
    laptop_cam = OpenCVCamera(laptop_config)
    wrist_cam = OpenCVCamera(wrist_config)

    try:
        laptop_cam.connect()
        print("✅ Laptop Camera Connected!")
        wrist_cam.connect()
        print("✅ Wrist Camera Connected!")

        # Allow cameras to warm up slightly/get ready
        time.sleep(1)

        # Read images
        print("📸 Reading images...")
        laptop_img = laptop_cam.read()
        wrist_img = wrist_cam.read()
        
        if laptop_img is None:
            print("❌ Failed to read from Laptop Camera")
        if wrist_img is None:
            print("❌ Failed to read from Wrist Camera")

        if laptop_img is not None and wrist_img is not None:
            # Combine images horizontally
            combined_img = np.hstack((laptop_img, wrist_img))
            
            # Save to final_victory.jpg (OpenCV uses BGR, read() returns RGB usually via LeRobot or BGR? 
            # LeRobot OpenCVCamera reads in BGR by default if using cv2 directly, but let's check config defaults.
            # actually OpenCVCameraConfig defaults to RGB color_mode.
            # verify color mode:
            # laptop_cam.read() returns RGB if color_mode is RGB.
            # cv2.imwrite expects BGR. So we convert if needed.
            
            # Let's assume standard behavior:
            # If config.color_mode is RGB (default), we get RGB.
            # cv2.imwrite needs BGR.
            combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite("final_victory.jpg", combined_img)
            print("✅ Saved combined snapshot to final_victory.jpg")

        # Read Current Joint Positions
        print("🦾 Reading Joint Positions...")
        # get_observation() returns a dictionary
        observation = robot.get_observation()
        
        # Filter for joint positions
        print("Current Joint Positions:")
        for key, value in observation.items():
            if "pos" in key:
                # Handle tensor or float
                val = value.item() if hasattr(value, "item") else value
                print(f"  {key}: {val}")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🧹 Cleaning up...")
        robot.disconnect()
        # Cameras disconnect
        try:
            laptop_cam.disconnect()
            wrist_cam.disconnect()
        except:
            pass
        print("✅ Done.")

if __name__ == "__main__":
    main()
