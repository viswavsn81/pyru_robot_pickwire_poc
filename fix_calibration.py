import json
import os

# 1. Define the correct motors for SO-100
motor_models = {
    "shoulder_pan": "xm430-w350",
    "shoulder_lift": "xm430-w350",
    "elbow_flex": "xm430-w350",
    "wrist_flex": "xl330-m288",
    "wrist_roll": "xl330-m288",
    "gripper": "xl330-m077"
}

# 2. Find the calibration file
# (Based on your previous logs, it is saved under 'None.json')
path = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json")

if not os.path.exists(path):
    print(f"❌ Error: Could not find calibration file at {path}")
    print("   Please run the calibration step again first!")
    exit(1)

# 3. Load, Patch, and Save
with open(path, "r") as f:
    data = json.load(f)

print(f"🔧 Patching file: {path}")
for motor_name, model_type in motor_models.items():
    if motor_name in data:
        data[motor_name]["model"] = model_type
        print(f"   - Set {motor_name} -> {model_type}")
    else:
        print(f"   ⚠️ Warning: {motor_name} missing from calibration!")

with open(path, "w") as f:
    json.dump(data, f, indent=4)

print("✅ Success! Calibration file repaired.")
