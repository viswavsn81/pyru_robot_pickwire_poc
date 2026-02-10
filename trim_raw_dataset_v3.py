import os
import glob
import json
import shutil
from pathlib import Path

SOURCE_DIR = Path("/home/pyru/lerobot/dataset")
DEST_DIR = Path("/home/pyru/lerobot/dataset_strict")

ARM_THRESHOLD = 80.0  # Must go above this to ARM the trigger
CUT_THRESHOLD = 75.0  # Must drop below this (after arming) to CUT

def main():
    print(f"🔪 Starting Raw Dataset Trim V3 (Numerical Sort)...")
    print(f"   Source: {SOURCE_DIR}")
    print(f"   Dest:   {DEST_DIR}")
    print(f"   Logic: Wait for > {ARM_THRESHOLD}, then Cut at < {CUT_THRESHOLD}")

    if DEST_DIR.exists():
        print(f"⚠️  Removing existing destination: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    # Find Episode Folders
    # episode_xxx sorting is also important but usually naturally okay with string sort unless > 999
    episode_folders = sorted(glob.glob(str(SOURCE_DIR / "episode_*")))
    
    if not episode_folders:
        print("❌ No episodes found in source directory.")
        return

    for ep_path in episode_folders:
        ep_name = os.path.basename(ep_path)
        src_ep_dir = Path(ep_path)
        dst_ep_dir = DEST_DIR / ep_name
        
        dst_ep_dir.mkdir(parents=True)
        
        # Get frame JSONs and SORT NUMERICALLY
        json_files = glob.glob(str(src_ep_dir / "frame_*.json"))
        
        # Critical Fix: Sort by frame ID (int) instead of string
        # frame_10.json should come after frame_2.json
        try:
            json_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
        except Exception as e:
            print(f"⚠️  {ep_name}: Error sorting files: {e}")
            continue

        if not json_files:
            print(f"⚠️  {ep_name}: No frame JSONs found.")
            continue
            
        print(f"{ep_name}: Processing {len(json_files)} files...")
            
        cutoff_index = -1
        is_armed = False
        min_val_after_arm = float('inf')
        
        # Analyze Frames
        for idx, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Robust Gripper Extraction
                gripper_val = None
                if 'action' in data:
                    if isinstance(data['action'], dict) and 'gripper' in data['action']:
                         gripper_val = data['action']['gripper']
                    elif isinstance(data['action'], (list, tuple)):
                        if len(data['action']) > 5:
                            gripper_val = data['action'][5]
                
                if gripper_val is not None:
                    # 1. Arming Check
                    if not is_armed:
                        if gripper_val > ARM_THRESHOLD:
                            is_armed = True
                    
                    # 2. Trigger Check (Only if Armed)
                    else:
                        if gripper_val < min_val_after_arm:
                            min_val_after_arm = gripper_val

                        if gripper_val < CUT_THRESHOLD:
                            cutoff_index = idx # This frame is the cut
                            break
                            
            except Exception as e:
                print(f"   Error reading {json_file}: {e}")
                
        # Determine Copy Range
        if cutoff_index != -1:
            frames_to_copy = cutoff_index + 1
            status = f"Trimmed at frame {cutoff_index}"
        else:
            frames_to_copy = len(json_files)
            if is_armed:
                status = "Armed but never Closed (Copied All)"
                print(f"      ⚠️  {ep_name}: Lowest value seen after arming: {min_val_after_arm} (Threshold: {CUT_THRESHOLD})")
            else:
                status = "Never Armed (Copied All)"
            
        # Copy Files
        copied_count = 0
        for i in range(frames_to_copy):
            src_json = str(json_files[i])
            basename = os.path.basename(src_json)
            frame_id = basename.replace("frame_", "").replace(".json", "")
            
            # Copy JSON
            shutil.copy(src_json, dst_ep_dir / basename)
            
            # Copy Images
            for cam in ["laptop", "wrist", "phone", "desk"]:
                img_name = f"{cam}_{frame_id}.jpg"
                src_img = src_ep_dir / img_name
                if src_img.exists():
                    shutil.copy(src_img, dst_ep_dir / img_name)
            
            copied_count += 1

        # Copy Meta Files
        meta_files = glob.glob(str(src_ep_dir / "*.json"))
        for m in meta_files:
            if "frame_" not in os.path.basename(m):
                shutil.copy(m, dst_ep_dir)
                
        other_meta = glob.glob(str(src_ep_dir / "*.csv"))
        for m in other_meta:
            shutil.copy(m, dst_ep_dir)

        print(f"✅ {ep_name}: Copied {copied_count} frames ({status})")

    print(f"\n🎉 Done! Strict dataset saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()
