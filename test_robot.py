import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Setup the configuration
# We use the specific config class found in your version
config = SOFollowerRobotConfig(port="/dev/ttyACM0")

# 2. Initialize the robot
print(f"Initializing robot on {config.port}...")
robot = SOFollower(config)

# 3. Connect (this turns Torque ON)
print("Connecting...")
robot.connect()

print("------------------------------------------------")
print("SUCCESS! Torque is ON.")
print("The robot should be stiff and holding its position.")
print("Waiting 5 seconds...")
print("------------------------------------------------")

time.sleep(5)

# 4. Disconnect (Torque OFF)
robot.disconnect()
print("Test complete. Torque OFF.")
