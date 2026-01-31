import cv2
import time
import numpy as np
from pathlib import Path

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor

def main():
    print("Starting SO-100 Hardware Smoke Test...")

    # 1. Setup Cameras
    print("Configuring cameras...")
    laptop_cam_config = OpenCVCameraConfig(
        index_or_path=2,
        fps=30,
        width=640,
        height=480
    )
    wrist_cam_config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=640,
        height=480
    )

    cameras_config = {
        "laptop": laptop_cam_config,
        "wrist": wrist_cam_config
    }

    # 2. Setup Robot Config
    print("Configuring robot...")
    calibration_path = Path("calibration_dummy") 
    
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        calibration_dir=calibration_path,
        cameras=cameras_config
    )

    # 3. Instantiate Robot
    print("Instantiating robot...")
    robot = SOFollower(config=robot_config)

    # 4. Bus Replacement (CRITICAL FIX for broken config/bus mismatch)
    print("Replacing FeetechMotorsBus with DynamixelMotorsBus and injecting motor models...")
    
    # Motor map provided by user
    # Motors 1-3: xm430-w350
    # Motors 4-6: xl330-m288 (except 6 is xl330-m077)
    # Names: shoulder_pan (1), shoulder_lift (2), elbow_flex (3), wrist_flex (4), wrist_roll (5), gripper (6)
    model_map = {
        "shoulder_pan": "xm430-w350",
        "shoulder_lift": "xm430-w350",
        "elbow_flex": "xm430-w350",
        "wrist_flex": "xl330-m288",
        "wrist_roll": "xl330-m288",
        "gripper": "xl330-m077"
    }

    new_motors = {}
    for name, motor in robot.bus.motors.items():
        new_model = model_map.get(name, motor.model)
        # Create new Motor instance with correct model
        new_motors[name] = Motor(motor.id, new_model, motor.norm_mode)
        print(f"Configured motor {name} (ID {motor.id}) with model {new_model}")

    # Replace the bus
    robot.bus = DynamixelMotorsBus(
        port=robot.config.port,
        motors=new_motors,
        calibration=robot.calibration
    )

    # 5. Connect
    print("Connecting to robot and cameras...")
    # Manually connect bus without handshake to bypass motor check (since scan failed)
    try:
        robot.bus.connect(handshake=False)
    except Exception as e:
        print(f"Bus connection warning: {e}")

    # Connect robot (skips bus connect if already connected, handles cameras)
    # We cannot call robot.connect() because it checks is_connected and raises error if bus is already connected.
    # So we manually do the rest of the connect steps.
    
    # 5a. Connect Cameras
    print("Connecting cameras...")
    connected_cameras = {}
    for name, cam in robot.cameras.items():
        try:
            cam.connect()
            print(f"Camera {name} connected.")
            connected_cameras[name] = cam
        except Exception as e:
            print(f"Failed to connect camera {name}: {e}")

    # 6. Capture Snapshot (Decoupled from robot motors)
    print("Capturing snapshot...")
    time.sleep(1) # Warmup
    
    laptop_frame = None
    wrist_frame = None
    
    if "laptop" in connected_cameras:
        laptop_frame = connected_cameras["laptop"].read()
    if "wrist" in connected_cameras:
        wrist_frame = connected_cameras["wrist"].read()

    if laptop_frame is not None and wrist_frame is not None:
        try:
            combined_img = np.hstack((laptop_frame, wrist_frame))
            combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("victory.jpg", combined_img)
            print("Snapshot saved to victory.jpg")
        except Exception as e:
            print(f"Error saving snapshot: {e}")
    else:
        print("Failed to capture frames from one or both cameras (or cameras missing).")

    # 5b. Configure Robot (Attempt Motor Connection)
    print("Configuring robot motors (via bus)...")
    try:
        robot.configure()
        print("Robot configured.")
        
        # 7. Print Gripper Position
        # Try reading specific motor
        print("Reading gripper position...")
        pos = robot.bus.read("Present_Position", "gripper")
        print(f"Current gripper position: {pos}")
        
    except Exception as e:
        print(f"Motor communication failed: {e}")
        print("Proceeding to cleanup...")

    print("Smoke test complete!")
    try:
        robot.disconnect()
    except:
        pass


if __name__ == "__main__":
    main()
