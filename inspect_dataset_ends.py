import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import glob

DATASET_DIR = Path("local/so100_test")

def main():
    print(f"Loading Dataset from: {DATASET_DIR}")

    # 1. Load Episode Metadata
    episode_files = sorted(glob.glob(str(DATASET_DIR / "meta/episodes/chunk-*/file-*.parquet")))
    if not episode_files:
        print("❌ No episode metadata found!")
        return
    
    print(f"Loading {len(episode_files)} episode files...")
    episodes_df = pd.concat([pd.read_parquet(f) for f in episode_files])
    
    # 2. Load Data (Actions)
    data_files = sorted(glob.glob(str(DATASET_DIR / "data/chunk-*/file-*.parquet")))
    if not data_files:
        print("❌ No data files found!")
        return
        
    print(f"Loading {len(data_files)} data files...")
    data_df = pd.concat([pd.read_parquet(f) for f in data_files])
    
    # Ensure sorted by index (raw LeRobot data usually has 'index' column)
    if 'index' in data_df.columns:
        data_df = data_df.sort_values('index').reset_index(drop=True)
    
    print(f"✅ Loaded {len(episodes_df)} episodes and {len(data_df)} frames.")

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Check for 'action' column. 
    # In parquet, 'action' might be a struct or list. 
    # Pandas read_parquet often keeps list columns as numpy arrays or lists.
    
    count = 0
    
    # Iterate episodes
    # We need to know where each episode starts/ends.
    # Episodes meta usually has 'length' or 'data_index' range.
    
    # Attempt to derive indices
    # If explicit index range exists:
    if 'data_index' in episodes_df.columns:
        # Assuming struct with 'from' and 'to'? No, usually not in pandas load unless flatted.
        # Let's check for 'length' and compute cumulative
        pass
    
    # Fallback: Use 'length' column if available to compute ranges
    if 'length' in episodes_df.columns:
        current_start = 0
        for i, row in episodes_df.iterrows():
            length = row['length']
            start = current_start
            end = start + length
            current_start = end
            
            # Slice Data
            episode_data = data_df.iloc[start:end]
            
            # Extract Action
            # If 'action' is a column of arrays
            if 'action' in episode_data.columns:
                actions = list(episode_data['action'])
            else:
                # If specific columns like 'action_0', 'action_1' etc exist (unlikely in LeRobot v3 which uses nested/list)
                print("Cannot find 'action' column.")
                break

            # Last 50 Frames
            if len(actions) > 50:
                last_50 = actions[-50:]
            else:
                last_50 = actions
                
            # Index 0 (Shoulder Pan)
            # actions is likely a list of arrays/lists
            pan_vals = [a[0] for a in last_50]
            
            plt.plot(pan_vals, alpha=0.6, linewidth=1.5, label=f"Ep {i}")
            count += 1
            
    else:
        print("❌ Could not determine episode lengths from metadata.")
        print("Columns:", episodes_df.columns)
        return

    plt.title("LAST 50 Frames: Shoulder Pan (Index 0)")
    plt.xlabel("Frame (Relative to End)")
    plt.ylabel("Angle (Degrees)")
    plt.grid(True, alpha=0.3)
    
    output_file = "dataset_bias.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")
    print("   Check image for drift.")

if __name__ == "__main__":
    main()
