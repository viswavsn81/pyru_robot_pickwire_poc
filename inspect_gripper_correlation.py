import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import glob
import random

DATASET_DIR = Path("local/so100_test")

def main():
    print(f"Loading Dataset from: {DATASET_DIR}")

    # 1. Load Episode Metadata
    episode_files = sorted(glob.glob(str(DATASET_DIR / "meta/episodes/chunk-*/file-*.parquet")))
    if not episode_files:
        print("❌ No episode metadata found!")
        return
    
    episodes_df = pd.concat([pd.read_parquet(f) for f in episode_files])
    
    # 2. Load Data (Actions)
    data_files = sorted(glob.glob(str(DATASET_DIR / "data/chunk-*/file-*.parquet")))
    if not data_files:
        print("❌ No data files found!")
        return
        
    data_df = pd.concat([pd.read_parquet(f) for f in data_files])
    
    if 'index' in data_df.columns:
        data_df = data_df.sort_values('index').reset_index(drop=True)
    
    print(f"✅ Loaded {len(episodes_df)} episodes and {len(data_df)} frames.")

    # 3. Select 5 Random Episodes
    total_episodes = len(episodes_df)
    indices = range(total_episodes)
    selected_indices = random.sample(indices, min(5, total_episodes))
    selected_indices.sort()
    
    print(f"Analyzing Episodes: {selected_indices}")

    # 4. Plotting
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 15), sharex=False)
    if len(selected_indices) == 1:
        axes = [axes]
    
    plt.subplots_adjust(hspace=0.4)

    # Compute episode ranges
    ranges = []
    if 'length' in episodes_df.columns:
        current_start = 0
        for i, row in episodes_df.iterrows():
            length = row['length']
            ranges.append((current_start, current_start + length))
            current_start += length
    else:
        print("❌ Missing 'length' column in metadata.")
        return

    # Plot Loop
    for plot_idx, ep_idx in enumerate(selected_indices):
        start, end = ranges[ep_idx]
        episode_data = data_df.iloc[start:end]
        
        if 'action' not in episode_data.columns:
            print("❌ 'action' column missing.")
            return

        actions = list(episode_data['action'])
        
        # Extract Joints: 0 (Pan) and 5 (Gripper)
        pan_vals = [a[0] for a in actions]
        grip_vals = [a[5] for a in actions]
        frames = range(len(actions))
        
        ax1 = axes[plot_idx]
        
        # Left Axis: Shoulder Pan (Blue)
        ax1.set_title(f"Episode {ep_idx} (Correlation Check)")
        ax1.plot(frames, pan_vals, 'b-', label='Shoulder Pan (J0)')
        ax1.set_ylabel('Shoulder Pan', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        # Right Axis: Gripper (Red Dashed)
        ax2 = ax1.twinx()
        ax2.plot(frames, grip_vals, 'r--', label='Gripper (J5)')
        ax2.set_ylabel('Gripper', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        if plot_idx == len(selected_indices) - 1:
            ax1.set_xlabel("Frame Number")

    output_file = "gripper_correlation.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")
    print("   Blue Line = Shoulder Pan (Left/Right Motion)")
    print("   Red Line = Gripper (Open/Close)")
    print("   LOOK FOR: Does Blue change just AFTER Red changes (drift after release)?")

if __name__ == "__main__":
    main()
