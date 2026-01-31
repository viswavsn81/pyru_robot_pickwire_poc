# SO-100 End-to-End Pipeline Guide
This authentic guide details the steps to record data, train a policy, and run it on the physical SO-100 robot.

## 1. Preparation
Ensure hardware is set up:
- **Robot**: SO-100 connected via USB (`/dev/ttyACM0`).
- **Cameras**:
    - Wrist Camera: Index 0
    - Laptop/External Camera: Index 2
- **Controller**: Xbox Controller connected via Bluetooth/USB.

## 2. Data Collection (Recording)
Run the recording script to collect teleoperation episodes.
```bash
python lerobot/record_dataset.py
```
### Controls
- **Start / Y**: Toggle Recording (Red circle appears on screen).
- **Back / X**: Exit script.
- **Joysticks**: Move robot arms.
- **A / B**: Close / Open Gripper.

**Data Location**: Episodes are saved to `dataset/episode_XXX`.

## 2.1. Clean Up (Optional)
To start collecting a brand new dataset from scratch:
```bash
rm -rf dataset/*             # Clear raw recordings
rm -rf local/so100_test      # Clear converted dataset
```
*Note: This permanently deletes previous data.*

## 2.2. Delete Last Episode
If you made a mistake (e.g., in the last recording), you can delete a specific episode:
```bash
rm -rf dataset/episode_XXX   # Replace XXX with the episode number (e.g., episode_008)
```

## 3. Data Conversion
Convert the raw recorded data into a LeRobot-compatible dataset format.
```bash
python lerobot/convert_data.py
```
- **Input**: Reads from `dataset/`.
- **Output**: Creates/Overwrites `local/so100_test`.
- **Modification**: To change the dataset name, edit `REPO_ID = "local/your_name"` in `convert_data.py`.

## 4. Policy Training
Train an ACT policy on the converted dataset.
```bash
python src/lerobot/scripts/lerobot_train.py \
    --policy.type=act \
    --dataset.repo_id=local/so100_test \
    --dataset.root=/home/pyru/lerobot/local/so100_test \
    --batch_size=10 \
    --steps=50000 \
    --policy.device=cuda \
    --job_name=so100_train_50ep \
    --policy.repo_id=local/so100_policy
```
- **batch_size**: Set to 10 for your 50-episode dataset.
- **steps**: Set to 50,000 for a comprehensive training run.

**Output**: Checkpoints are saved in `outputs/train/{date}/{job_name}/checkpoints`.

## 5. Deployment (Autonomous Run)
Run the trained policy on the robot.

1. **Locate Checkpoint**: Find the path to your best checkpoint (e.g., `outputs/train/.../checkpoints/000500/pretrained_model`).
2. **Update Script**: Edit `lerobot/autonomous_run.py`:
   ```python
   CHECKPOINT_PATH = Path("/path/to/your/checkpoint/pretrained_model")
   ```
3. **Run**:
   ```bash
   python lerobot/autonomous_run.py
   ```
- **Stop**: Press `Ctrl+C` to safely stop the robot.
