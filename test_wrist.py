import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Connect
config = SOFollowerRobotConfig(port="/dev/ttyACM0")
robot = SOFollower(config)
robot.connect()

motor_name = "wrist_flex"
print(f"Checking {motor_name}...")

# 2. Read Position
try:
    start_pos = robot.bus.read("Present_Position", motor_name)
    print(f"Current Wrist Position: {start_pos}")
except Exception as e:
    print(f"ERROR: Could not read wrist! Check the cable. Details: {e}")
    robot.disconnect()
    exit()

# 3. The "Big Move" (200 units is about 20 degrees - very visible)
print("Commanding LARGE wrist movement...")

target_up = int(start_pos + 200)
target_down = int(start_pos - 200)
center = int(start_pos)

try:
    for i in range(3):
        print(f"Flex Up {i+1}...")
        robot.bus.write("Goal_Position", motor_name, target_up)
        time.sleep(0.5)

        print(f"Flex Down {i+1}...")
        robot.bus.write("Goal_Position", motor_name, target_down)
        time.sleep(0.5)

    # Reset
    print("Centering...")
    robot.bus.write("Goal_Position", motor_name, center)
    time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopping...")

robot.disconnect()
print("Done.")
