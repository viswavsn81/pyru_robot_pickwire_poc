import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

# 1. Connect
config = SOFollowerRobotConfig(port="/dev/ttyACM0")
robot = SOFollower(config)
robot.connect()

print("Robot connected. Starting FIXED Pick & Place Demo...")

def move(motor, value_change, duration=0.5):
    current = robot.bus.read("Present_Position", motor)
    target = int(current + value_change)
    robot.bus.write("Goal_Position", motor, target)
    time.sleep(duration)

try:
    # --- PHASE 1: PREPARE ---
    print("1. Opening Gripper...")
    move("gripper", -200) 
    
    # --- PHASE 2: REACH DOWN ---
    print("2. Reaching down...")
    move("shoulder_lift", -150) # Tilt shoulder forward
    
    # *** FIXED JOINT 3 (ELBOW) ***
    # We flipped this from -100 to +100 so it extends OUT instead of curling IN
    move("elbow_flex", 100)    
    
    move("wrist_flex", -200)    # Angle wrist down
    
    time.sleep(1)

    # --- PHASE 3: GRAB ---
    print("3. GRABBING object...")
    move("gripper", 300)        # Close tight
    time.sleep(0.5)

    # --- PHASE 4: LIFT ---
    print("4. Lifting up...")
    move("shoulder_lift", 150)  # Tilt shoulder back
    
    # *** FIXED JOINT 3 (ELBOW) ***
    # We flipped this from +100 to -100 to make it retract correctly
    move("elbow_flex", -100)     
    
    move("wrist_flex", 200)     # Level wrist
    
    time.sleep(1)

    # --- PHASE 5: INSPECT ---
    print("5. Inspecting...")
    move("wrist_roll", 300, 1.0)
    move("wrist_roll", -600, 1.0)
    move("wrist_roll", 300, 1.0)

    print("Demo Complete!")

except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error: {e}")

robot.disconnect()
print("Done.")
