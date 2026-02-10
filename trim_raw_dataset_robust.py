import os
import glob
import json
import shutil
from pathlib import Path

SOURCE_DIR = Path("/home/pyru/lerobot/dataset")
DEST_DIR = Path("/home/pyru/lerobot/dataset_strict")

# THRESHOLDS
ARM_THRESHOLD = 60.0  # Gripper must go ABOVE this to "Arm"
CUT_THRESHOLD = 75.0  # Gripper must drop BELOW this to "Cut" (Once Armed)

def main():
    print(f"🔪 Starting Robust Trim (No Skipping + Strict Sort)...")
    print(f"   Logic: Wait for > {ARM_THRESHOLD}, then Cut at < {CUT_THRESHOLD}")

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    # Find Episode Folders
    episode_folders = sorted(glob.glob(str(SOURCE_DIR / "episode_*")))

    for ep_path in episode_folders:
        ep_name = os.path.basename(ep_path)
        src_ep_dir = Path(ep_path)
        dst_ep_dir = DEST_DIR / ep_name
        
        dst_ep_dir.mkdir(parents=True)
        
        # 1. GET FILES & SORT NUMERICALLY (CRITICAL FIX)
        json_files = glob.glob(str(src_ep_dir / "frame_*.json"))
        try:
            json_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
        except Exception as e:
            print(f"⚠️  {ep_name}: Sorting Error {e}")
            continue

        # 2. ANALYZE (State Machine)
        cutoff_index = -1
        is_armed = False
        min_val_seen = 1000.0
        
        for idx, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Robust Value Extraction
                val = None
                if 'action' in data:
                    if isinstance(data['action'], dict) and 'gripper' in data['action']:
                         val = data['action']['gripper']
                    elif isinstance(data['action'], (list, tuple)) and len(data['action']) > 5:
                        val = data['action'][5]
                
                if val is not None:
                    if val < min_val_seen: min_val_seen = val
                    
                    # State Machine
                    if not is_armed:
                        if val > ARM_THRESHOLD:
                            is_armed = True
                    else:
                        if val < CUT_THRESHOLD:
                            cutoff_index = idx
                            print(f"   ✂️ Cut triggered at frame {idx} (Val: {val:.2f})")
                            break
            except:
                pass
                
        # 3. DETERMINE COUNT
        if cutoff_index != -1:
            count = cutoff_index + 1
            status = f"Trimmed at {cutoff_index}"
        else:
            count = len(json_files)
            status = f"Kept All (Min Val: {min_val_seen:.2f})"
            
        # 4. COPY FILES
        # print(f"   Copying {count} frames...")
        for i in range(count):
            src = json_files[i]
            frame_id = os.path.basename(src).split('_')[1].split('.')[0]
            
            # Copy JSON
            shutil.copy(src, dst_ep_dir)
            
            # Copy Images
            for cam in ['laptop', 'wrist', 'phone']:
                img_name = f"{cam}_{frame_id}.jpg"
                src_img = src_ep_dir / img_name
                if src_img.exists():
                    shutil.copy(src_img, dst_ep_dir)
                    
        # Copy Metadata
        for meta in glob.glob(str(src_ep_dir / "*.json")):
            if "frame_" not in meta: shutil.copy(meta, dst_ep_dir)
        for meta in glob.glob(str(src_ep_dir / "*.csv")): shutil.copy(meta, dst_ep_dir)
        
        print(f"✅ {ep_name}: {status}")

    print(f"\n🎉 Done! Saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()
