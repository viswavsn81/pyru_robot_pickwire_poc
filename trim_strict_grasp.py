import pandas as pd
import numpy as np
import shutil
import glob
from pathlib import Path
import json

INPUT_DIR = Path("local/so100_test")
OUTPUT_DIR = Path("local/so100_test_strict")
GRIPPER_THRESHOLD = -10.0 # Below this = closed

def main():
    if OUTPUT_DIR.exists():
        print(f"⚠️  Output directory {OUTPUT_DIR} exists. Removing...")
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "data/chunk-000").mkdir(parents=True)
    (OUTPUT_DIR / "meta/episodes/chunk-000").mkdir(parents=True)

    print(f"Loading from: {INPUT_DIR}")
    
    # 1. Load Original Metadata
    ep_files = sorted(glob.glob(str(INPUT_DIR / "meta/episodes/chunk-*/file-*.parquet")))
    episodes_df = pd.concat([pd.read_parquet(f) for f in ep_files])
    
    # 2. Load Original Data
    data_files = sorted(glob.glob(str(INPUT_DIR / "data/chunk-*/file-*.parquet")))
    data_df = pd.concat([pd.read_parquet(f) for f in data_files])

    if 'index' in data_df.columns:
        data_df = data_df.sort_values('index').reset_index(drop=True)

    print(f"Loaded {len(episodes_df)} episodes, {len(data_df)} frames.")
    
    # 3. Process Episodes
    new_episodes = []
    new_data_frames = []
    
    global_frame_idx = 0
    
    sample_stats = []

    # Iterate over existing episodes
    start_idx = 0
    for i, row in episodes_df.iterrows():
        length = row['length']
        end_idx = start_idx + length
        
        # Slice Episode Data
        ep_data = data_df.iloc[start_idx:end_idx].copy()
        original_len = len(ep_data)
        
        # Find Trigger (First frame where gripper < threshold)
        # Action is usually list or array
        actions = list(ep_data['action'])
        cut_idx = -1
        
        for frame_i, act in enumerate(actions):
            gripper_val = act[5] # Index 5
            if gripper_val < GRIPPER_THRESHOLD:
                cut_idx = frame_i
                break
        
        if cut_idx != -1:
            # Keep up to cut_idx (inclusive or exclusive? User said "Discard everything AFTER... where it closes")
            # Usually we want the frame where it closes to be the last one, or slightly before.
            # User said: "Slice... to Trigger_Frame."
            # Let's include the trigger frame so the policy sees the command to close.
            final_len = cut_idx + 1
            ep_data = ep_data.iloc[:final_len]
        else:
            final_len = original_len
            # No trim happened
        
        # Update Data Indices for new dataset
        # We need to rebuild the 'index' column and handle concatenation
        # Actually LeRobot dataset usually requires 'index', 'timestamp', 'episode_index', 'frame_index'
        
        # Update episode_index and frame_index checks
        ep_data['episode_index'] = i
        ep_data['frame_index'] = np.arange(final_len)
        ep_data['index'] = np.arange(global_frame_idx, global_frame_idx + final_len)
        if 'timestamp' in ep_data.columns:
            # Recalculate timestamps? Usually just shift start to 0?
            # Or keep delta? Assuming 30FPS usually.
            # Let's trust existing timestamps are 0-based or relative. 
            # If they are relative to episode start, valid.
            pass

        new_data_frames.append(ep_data)
        
        # New Episode Metadata
        new_ep = row.copy()
        new_ep['length'] = final_len
        new_ep['index'] = i 
        # Note: 'index' in episodes_df usually refers to episode index
        new_episodes.append(new_ep)
        
        if i < 3: # Capture stats for first 3
            sample_stats.append(f"Ep {i}: {original_len} -> {final_len} frames")

        global_frame_idx += final_len
        start_idx += length # Advance original index

    # 4. Construct New Dataframes
    new_episodes_df = pd.DataFrame(new_episodes)
    new_data_df = pd.concat(new_data_frames)
    
    print(f"✅ Trimmed Dataset: {len(new_episodes_df)} episodes, {len(new_data_df)} frames.")
    for stat in sample_stats:
        print(f"   {stat}")

    # 5. Save Parquet
    print("Saving to disk...")
    new_episodes_df.to_parquet(OUTPUT_DIR / "meta/episodes/chunk-000/file-000.parquet")
    new_data_df.to_parquet(OUTPUT_DIR / "data/chunk-000/file-000.parquet")
    
    # 6. Copy Configs
    for f in ["info.json", "stats.json", "tasks.parquet"]:
        src = INPUT_DIR / "meta" / f
        if src.exists():
            shutil.copy(src, OUTPUT_DIR / "meta" / f)
            
    print(f"✅ Success! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import numpy as np # Ensure accessible if needed by eval
    main()
