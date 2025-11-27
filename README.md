# ArmRL - RL Training for Robotic Manipulation

Train RL agents (PPO, SAC, TD3) for robotic manipulation tasks using Robosuite and stable-baselines3. Supports both **camera-based** (visual) and **object state-based** (proprioceptive) observations.

**Supported Algorithms:** PPO | SAC | TD3
**Observation Types:** Camera RGB | Object States

## Features

- 🎯 Multiple RL algorithms (PPO, SAC, TD3)
- 📷 Camera-based training with CNN policies
- 🤖 Object state-based training with MLP policies
- 🎬 Automatic video recording for both observation types
- ⚙️ YAML-based configuration system
- 📊 TensorBoard logging
- 🔄 Automatic config file backup to model directory
- 💾 Checkpoint and best model saving

## Installation

```bash
conda create -n armrl python=3.12 -y
conda activate armrl
pip install mujoco robosuite "imageio[ffmpeg]" opencv-python PyYAML
pip install gymnasium stable-baselines3[extra] tensorboard h5py
```

## Quick Start

### Training

```bash
# Train with camera observations (image-based)
python train.py --config configs/fast.yaml

# Train with object state observations (proprioceptive)
python train.py --config configs/baseline.yaml

# High quality camera-based training
python train.py --config configs/high_quality.yaml

# CPU-only training
python train.py --config configs/cpu_only.yaml
```

### Testing

```bash
# Test a trained model (uses config file)
python test_model.py --config configs/baseline.yaml

# The model path is specified in the config file under test.model_path
```

## Configuration Files

Training is configured via YAML files in the `configs/` directory:

### Available Configs

| Config File | Observation Type | Policy | Description |
|-------------|------------------|--------|-------------|
| `configs/fast.yaml` | Camera (RGB) | CnnPolicy | Quick experiments (64x64 images, 16 envs) |
| `configs/high_quality.yaml` | Camera (RGB) | CnnPolicy | Best results (128x128 images, 1M steps) |
| `configs/baseline.yaml` | Object States | MlpPolicy | State-based baseline (no camera) |
| `configs/baseline_no_reward_shape.yaml` | Object States | MlpPolicy | Baseline without reward shaping |
| `configs/cpu_only.yaml` | Camera (RGB) | CnnPolicy | CPU-only training |
| `configs/test_rgb.yaml` | Camera (RGB) | CnnPolicy | RGB testing config |
| `configs/test_state.yaml` | Object States | MlpPolicy | State testing config |

### Configuration Structure

```yaml
env:
  env_name: "PickPlaceCan"
  robots: ["Panda"]
  horizon: 200
  use_camera_obs: true     # false for object state observations
  use_object_obs: false    # true for object state observations
  camera_height: 84        # only used if use_camera_obs=true
  camera_width: 84
  has_renderer: false
  has_offscreen_renderer: true  # needed for video recording
  reward_shaping: true

alg:
  alg_name: "PPO"
  policy: "CnnPolicy"      # "MlpPolicy" for object state observations
  learning_rate: 0.0003
  n_steps: 512
  batch_size: 64
  n_epochs: 10
  use_sde: true            # State Dependent Exploration
  sde_sample_freq: 8

train:
  n_envs: 16
  device: "cuda"
  total_timesteps: 500000
  save_freq: 20000         # Save checkpoint every N steps
  eval_freq: 10000         # Evaluate every N steps
  model_save_path: "./models/"
  log_path: "./logs/"
  tensorboard_log: "./tensorboard_logs/"

test:
  model_path: "./models/best_model/best_model.zip"
  n_episodes: 5
  deterministic: true
  save_video: true
  result_path: "./videos/"
  video_fps: 20
```

## Observation Types

### Camera Observations (Image-based)
- Uses RGB images from camera
- Requires `use_camera_obs: true` in config
- Uses `CnnPolicy` to process visual input
- Automatically transposes images for PyTorch CNNs
- Example: [configs/fast.yaml](configs/fast.yaml)

### Object State Observations (Proprioceptive)
- Uses robot joint angles, positions, object positions, etc.
- Requires `use_object_obs: true` and `use_camera_obs: false` in config
- Uses `MlpPolicy` to process state vectors
- Can still record videos during testing
- Example: [configs/baseline.yaml](configs/baseline.yaml)

## Algorithm Comparison

| Algorithm | Best For | Parallel Envs | Sample Efficiency | Stability |
|-----------|----------|---------------|-------------------|-----------|
| **PPO** | General purpose, beginners | 8-32 | Medium | High |
| **SAC** | Sample efficiency | 4-8 | High | High |
| **TD3** | Deterministic control | 4-8 | High | Medium |

## Training Features

### Automatic Config Backup
After training completes, the configuration file is automatically copied to the model directory for reproducibility:
```
./models/
├── best_model/
│   └── best_model.zip
├── fast.yaml  # Automatically copied
└── PPO_PickPlaceCan_20000_steps.zip
```

### Video Recording
Both camera-based and object state-based policies can save videos during testing:
- Camera policies: Uses the camera observation
- Object state policies: Renders the environment camera view
- Videos are saved to the path specified in `test.result_path`

### Callbacks
- **CheckpointCallback**: Saves model checkpoints at regular intervals
- **EvalCallback**: Evaluates the model periodically and saves the best model

## Monitor Training

```bash
# View training metrics in TensorBoard
tensorboard --logdir ./tensorboard_logs/

# Monitor training progress
watch -n 5 'ls -lh ./models/'
```

## Running in Background

### Using nohup with live output:
```bash
nohup python train.py --config configs/baseline.yaml > train.log 2>&1 &
tail -f train.log
```

### Using screen (recommended):
```bash
# Start a screen session
screen -S training

# Run training
python train.py --config configs/baseline.yaml

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Using tmux:
```bash
# Start a tmux session
tmux new -s training

# Run training
python train.py --config configs/baseline.yaml

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## Project Structure

```
ArmRL/
├── train.py              # Main training script
├── test_model.py         # Test trained models and save videos
├── config.py             # Configuration classes and YAML loader
├── env_wrapper.py        # Gymnasium wrappers for Robosuite
├── configs/              # YAML configuration files
│   ├── fast.yaml         # Quick camera-based training
│   ├── baseline.yaml     # Object state baseline
│   ├── high_quality.yaml # High quality camera training
│   └── ...
├── models/               # Saved models (created during training)
├── logs/                 # Training logs
├── tensorboard_logs/     # TensorBoard logs
└── videos/               # Test videos (created during testing)
```

## Key Implementation Details

### VecTransposeImage Wrapper
- Only applied when using camera observations
- Converts image format from (H, W, C) to (C, H, W) for PyTorch CNNs
- Automatically skipped for object state observations

### Environment Creation
- Camera observations: Uses `RobosuiteImageWrapper`
- Object observations: Uses `GymWrapper` directly
- Both support parallel environments via `SubprocVecEnv`

### Video Recording
- Forces offscreen renderer during testing
- Renders camera view even for object state policies
- Saves videos as MP4 with configurable FPS

## Troubleshooting

### "axes don't match array" error
This occurs when using camera observations config with object state policy or vice versa. Ensure:
- Camera obs: `use_camera_obs: true`, `policy: "CnnPolicy"`
- Object obs: `use_object_obs: true`, `use_camera_obs: false`, `policy: "MlpPolicy"`

### Out of memory
- Reduce `n_envs` (parallel environments)
- Use smaller image size (e.g., 64x64 instead of 128x128)
- Reduce `batch_size`

### Slow training
- Increase `n_envs` for faster data collection
- Use GPU: `device: "cuda"`
- Use smaller images for camera-based training

## Example Workflow

```bash
# 1. Train a camera-based policy
python train.py --config configs/fast.yaml

# 2. Monitor training
tensorboard --logdir ./tensorboard_logs/

# 3. Test the trained model
python test_model.py --config configs/fast.yaml

# 4. Compare with object state baseline
python train.py --config configs/baseline.yaml
python test_model.py --config configs/baseline.yaml
```

## Citation

This project uses:
- [Robosuite](https://robosuite.ai/) - Robot simulation environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [MuJoCo](https://mujoco.org/) - Physics engine
