import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Setup & Connect
config = SOFollowerRobotConfig(port="/dev/ttyACM0")
robot = SOFollower(config)
robot.connect()

print("Robot connected.")

# 2. Select just ONE motor to test
motor_name = "shoulder_pan"

# 3. Read the starting position
start_pos = robot.bus.read("Present_Position", motor_name)
print(f"Shoulder Start Position: {start_pos}")

print("Wiggling shoulder...")

# We use a small movement (20 units) to be safe
# Note the order: write(Register, Motor_Name, Value)
target_up = int(start_pos + 20)
target_down = int(start_pos - 20)
target_center = int(start_pos)

try:
    for i in range(3):
        print(f"Wave {i+1}/3")

        # Move Left
        robot.bus.write("Goal_Position", motor_name, target_up)
        time.sleep(0.5)

        # Move Right
        robot.bus.write("Goal_Position", motor_name, target_down)
        time.sleep(0.5)

    # Return to start
    print("Returning to center...")
    robot.bus.write("Goal_Position", motor_name, target_center)
    time.sleep(1.0)

except KeyboardInterrupt:
    print("Stopping...")

robot.disconnect()
print("Done! Torque OFF.")
