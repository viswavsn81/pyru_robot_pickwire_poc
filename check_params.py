import sys
import os
# Force python to look in your local src folder
sys.path.append(os.path.abspath("src"))

try:
    from lerobot.common.robot_devices.cameras.opencv import OpenCVCameraConfig
    print("\n✅ SUCCESS! Here are the valid fields for your cameras:")
    print(list(OpenCVCameraConfig.__dataclass_fields__.keys()))
    print("\n")
except Exception as e:
    print(f"\n❌ Error inspecting config: {e}\n")
