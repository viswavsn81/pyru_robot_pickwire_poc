import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Connect
config = SOFollowerRobotConfig(port="/dev/ttyACM0")
robot = SOFollower(config)
robot.connect()

# TARGET: Joint 5 (Wrist Roll)
motor_name = "wrist_roll"
print(f"Connecting to Joint 5 ({motor_name})...")

try:
    # 2. Read Position
    start_pos = robot.bus.read("Present_Position", motor_name)
    print(f"Joint 5 Start Position: {start_pos}")

    print("Rotating Wrist Left and Right...")

    # We will rotate it +/- 150 units (about 15 degrees)
    target_cw = int(start_pos + 150)
    target_ccw = int(start_pos - 150)
    center = int(start_pos)

    for i in range(3):
        print(f"Rotate {i+1}...")
        # Clockwise
        robot.bus.write("Goal_Position", motor_name, target_cw)
        time.sleep(0.5)

        # Counter-Clockwise
        robot.bus.write("Goal_Position", motor_name, target_ccw)
        time.sleep(0.5)

    # Reset
    print("Centering...")
    robot.bus.write("Goal_Position", motor_name, center)
    time.sleep(0.5)

except Exception as e:
    print(f"ERROR: Could not move Joint 5. Details: {e}")

robot.disconnect()
print("Test Complete.")
