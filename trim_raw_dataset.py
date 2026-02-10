import os
import glob
import json
import shutil
from pathlib import Path

SOURCE_DIR = Path("/home/pyru/lerobot/dataset")
DEST_DIR = Path("/home/pyru/lerobot/dataset_strict")
GRIPPER_THRESHOLD = 40.0 # Below this = Closing

def main():
    print(f"🔪 Starting Raw Dataset Trim...")
    print(f"   Source: {SOURCE_DIR}")
    print(f"   Dest:   {DEST_DIR}")
    print(f"   Threshold: < {GRIPPER_THRESHOLD}")

    if DEST_DIR.exists():
        print(f"⚠️  Removing existing destination: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    # Find Episode Folders
    episode_folders = sorted(glob.glob(str(SOURCE_DIR / "episode_*")))
    
    if not episode_folders:
        print("❌ No episodes found in source directory.")
        return

    for ep_path in episode_folders:
        ep_name = os.path.basename(ep_path)
        src_ep_dir = Path(ep_path)
        dst_ep_dir = DEST_DIR / ep_name
        
        dst_ep_dir.mkdir(parents=True)
        
        # Get all frame JSONs
        json_files = sorted(glob.glob(str(src_ep_dir / "frame_*.json")))
        
        if not json_files:
            print(f"⚠️  {ep_name}: No frame JSONs found.")
            continue
            
        cutoff_index = -1
        
        # Analyze Frames
        for idx, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Check Gripper Value
                # User specified: data['action']['gripper']
                # Be robust if structure varies slightly
                gripper_val = None
                if 'action' in data:
                    if isinstance(data['action'], dict) and 'gripper' in data['action']:
                         gripper_val = data['action']['gripper']
                    elif isinstance(data['action'], (list, tuple)):
                        # Fallback if action is list (Index 5 usually gripper)
                        if len(data['action']) > 5:
                            gripper_val = data['action'][5]
                
                if gripper_val is not None:
                    if gripper_val < GRIPPER_THRESHOLD:
                        cutoff_index = idx
                        break
            except Exception as e:
                print(f"   Error reading {json_file}: {e}")
                
        # Determine Copy Range
        if cutoff_index != -1:
            # Copy up to and including the cutoff frame
            frames_to_copy = cutoff_index + 1
            status = f"Trimmed at frame {cutoff_index}"
        else:
            # Copy all
            frames_to_copy = len(json_files)
            status = "No Cutoff Found (Copied All)"
            
        # Copy Files
        copied_count = 0
        for i in range(frames_to_copy):
            # Define filenames
            # Assuming format frame_000000.json
            # But relying on sorted globe order matches index i is risky if gaps exist?
            # Usually record_dataset saves sequential 000000, 000001
            # Let's use the actual file found at index i in the sorted list
            
            src_json = str(json_files[i])
            basename = os.path.basename(src_json) # frame_xxxxxx.json
            frame_id = basename.replace("frame_", "").replace(".json", "")
            
            # Copy JSON
            shutil.copy(src_json, dst_ep_dir / basename)
            
            # Copy Images
            # laptop_xxxxxx.jpg and wrist_xxxxxx.jpg (or phone_xxxxxx.jpg?)
            # User specified: "laptop_*.jpg, wrist_*.jpg"
            for cam in ["laptop", "wrist", "phone", "desk"]: # Check common names
                img_name = f"{cam}_{frame_id}.jpg"
                src_img = src_ep_dir / img_name
                if src_img.exists():
                    shutil.copy(src_img, dst_ep_dir / img_name)
            
            copied_count += 1

        # Copy Meta Files (stats.json, etc)
        meta_files = glob.glob(str(src_ep_dir / "*.json"))
        for m in meta_files:
            if "frame_" not in os.path.basename(m):
                shutil.copy(m, dst_ep_dir)
                
        # Copy any CSVs or other metadata
        other_meta = glob.glob(str(src_ep_dir / "*.csv"))
        for m in other_meta:
            shutil.copy(m, dst_ep_dir)

        print(f"✅ {ep_name}: Copied {copied_count} frames ({status})")

    print(f"\n🎉 Done! Strict dataset saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()
