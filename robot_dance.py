import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Setup
config = SOFollowerRobotConfig(port="/dev/ttyACM0")
robot = SOFollower(config)
robot.connect()

print("Robot connected. Starting full system check...")

# List of all motors in the order we want to test them
motor_list = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'gripper']

try:
    for motor_name in motor_list:
        print(f"Testing: {motor_name}...")

        # Read current position
        current_pos = robot.bus.read("Present_Position", motor_name)

        # Define targets (Small wiggle of +/- 50 units)
        # 50 units is roughly 4 degrees, safe for all joints
        target_a = int(current_pos + 50)
        target_b = int(current_pos - 50)
        center   = int(current_pos)

        # Wiggle
        robot.bus.write("Goal_Position", motor_name, target_a)
        time.sleep(0.3)
        robot.bus.write("Goal_Position", motor_name, target_b)
        time.sleep(0.3)

        # Return to center
        robot.bus.write("Goal_Position", motor_name, center)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopping...")

robot.disconnect()
print("Full System Check Complete!")
