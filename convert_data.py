#!/usr/bin/env python

import json
from pathlib import Path
import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
REPO_ID = "local/so100_test"
FPS = 30
DATASET_ROOT = Path("dataset_strict") # Pointing to the trimmed raw data

# Define features
FEATURES = {
    "observation.images.laptop": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
}

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

def main():
    # 1. Create Empty Dataset
    print(f"🚀 Creating LeRobotDataset: {REPO_ID}")
    
    dataset_out = Path("local/so100_test")
    if dataset_out.exists():
        import shutil
        shutil.rmtree(dataset_out)

    try:
        ds = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            features=FEATURES,
            robot_type="so100",
            root=dataset_out
        )
    except FileExistsError:
        print(f"⚠️  Dataset {REPO_ID} already exists locally. Please remove it or change REPO_ID.")
        return

    # 2. Iterate through episodes
    if not DATASET_ROOT.exists():
        print(f"❌ Source directory {DATASET_ROOT} does not exist!")
        return

    episode_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    
    if not episode_dirs:
        print(f"❌ No episodes found in {DATASET_ROOT} folder.")
        return

    print(f"📂 Found {len(episode_dirs)} episodes in {DATASET_ROOT}.")

    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"  Processing {ep_dir.name} ({ep_idx + 1}/{len(episode_dirs)})...")
        
        # Load all frames for this episode
        # Use numerical sort for frame loading too!
        import os
        frame_files = sorted(ep_dir.glob("frame_*.json"), key=lambda f: int(f.stem.split('_')[1]))
        
        if not frame_files:
            print(f"  ⚠️  No frames found in {ep_dir}")
            continue

        for frame_file in frame_files:
            frame_idx_str = frame_file.stem.split("_")[1]
            
            with open(frame_file, "r") as f:
                data = json.load(f)
            
            # Load Images
            laptop_path = ep_dir / f"laptop_{frame_idx_str}.jpg"
            wrist_path = ep_dir / f"wrist_{frame_idx_str}.jpg"
            
            if not laptop_path.exists() or not wrist_path.exists():
                print(f"    ❌ Missing images for frame {frame_idx_str}")
                continue

            laptop_img = Image.open(laptop_path)
            wrist_img = Image.open(wrist_path)
            
            # Prepare State and Action vectors
            state_vec = []
            action_vec = []
            
            obs_data = data["observation"]
            act_data = data["action"]
            
            for name in MOTOR_NAMES:
                state_vec.append(obs_data.get(f"{name}.pos", 0.0))
                # Allow for list-based actions if JSON has them?
                # Your code snippet assumed dict get(name, 0.0) which is fine for raw recorded data
                # But robustness check:
                val = act_data.get(name, 0.0)
                action_vec.append(val)
                
            # Create Frame Dictionary
            frame = {
                "observation.images.laptop": laptop_img,
                "observation.images.wrist": wrist_img,
                "observation.state": torch.tensor(state_vec, dtype=torch.float32),
                "action": torch.tensor(action_vec, dtype=torch.float32),
                "task": "Teleoperation",
            }
            
            ds.add_frame(frame)
        
        # Save Episode
        ds.save_episode(episode_data=None) 
        print(f"  ✅ Saved {ep_dir.name}")

    # 3. Finalize
    print("💾 Finalizing dataset...")
    ds.finalize()
    print("✅ Conversion Complete! Dataset saved to:", ds.root)

if __name__ == "__main__":
    main()
