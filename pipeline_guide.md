# SO-100 Hybrid Pipeline Guide

This guide documents the exact commands for the "Hybrid" workflow (Stop-Motion + Real-Time Clutch) using the StereoPi Camera via FFmpeg and a USB Desk Camera.

## 1. Recording Data (Hybrid 3D Stop-Motion)

Run the recorder script which uses **FFmpeg** for the wrist camera (UDP) and **OpenCV** for the desk camera.

**Command:**
```bash
python record_dataset_3d_stopmotion_hybrid.py
```

**Controls:**
- **Hold RT (Trigger)**: Manual/Limp Mode (Hand-over-hand positioning).
- **Sticks**: Teleoperation.
- **Start (Menu)**: Start/Stop Episode.
- **X Button**: Switch to "Stop-Motion" mode (Single frame capture).
- **Y Button**: Capture frame (when in Stop-Motion mode).

---

## 2. Training (ACT Policy - Low Memory)

Use this specific command to train the **ACT** policy on a GPU with limited VRAM (e.g., laptop GPU). Typical training time: ~2000 steps.

**Command:**
```bash
python src/lerobot/scripts/lerobot_train.py \
  --policy.type=act \
  --dataset.repo_id=local/so100_test \
  --dataset.root=dataset \
  --batch_size=16 \
  --steps=2000 \
  --save_freq=1000 \
  --policy.use_amp=true \
  --policy.device=cuda \
  --job_name=act_hybrid_run
```

**Key Parameters:**
- `batch_size=16`: Reduced from default to fit VRAM.
- `policy.use_amp=true`: Enables Automatic Mixed Precision (saves memory & speeds up training).
- `steps=2000`: Sufficient for a quick robust policy.

---

## 3. Inference (Running the Policy)

Run the inference script on the real robot.

**Command:**
```bash
python run_ACT_policy_hybrid.py
```

**Notes:**
- **Key Mapping Fix**: This script handles the 'laptop' vs 'desk' key mapping automatically (Training expects 'laptop', Recorder saves 'laptop' (as desk), so it matches).
- **Safety**: Ensure you have a hand on the kill switch (Ctrl+C) or Spacebar (if safety switch enabled).

---

## 4. Data Management

Commands to manage your dataset.

**Delete a Specific Episode (e.g., episode_005):**
```bash
rm -rf dataset/episode_005
```

**Delete ALL Raw Episodes (Clear Dataset):**
```bash
rm -rf dataset/episode_*
```

**Delete Converted Data (LeRobot Format):**
```bash
rm -rf local/so100_test
```

**Full Reset (Delete EVERYTHING):**
```bash
rm -rf dataset/episode_* local/so100_test
```
