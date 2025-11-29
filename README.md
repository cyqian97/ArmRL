# ArmRL - RL Training for Robotic Manipulation

Train RL agents (PPO, SAC, TD3) for robotic manipulation tasks using Robosuite and stable-baselines3. Supports both **camera-based** (visual) and **object state-based** (proprioceptive) observations.

**Supported Algorithms:** PPO | SAC | TD3
**Observation Types:** Camera RGB | Object States

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
# Train with the default config (image-based)
python train.py --config configs/default.yaml
```

### Testing

```bash
# Test with the default config (image-based)
python test_model.py --config configs/default.yaml

# The model path is specified in the config file under test.model_path
```

## Configuration Files

Training is configured via YAML files in the `configs/` directory:

### Configuration Structure

```yaml
env:
  env_name: "Lift"                  # Environment name (e.g., "Lift", "PickPlaceCan")
  robots: ["Panda"]                 # Robot types
  horizon: 200                      # Episode length
  control_freq: 20                  # Control frequency (Hz)
  use_camera_obs: true              # Use camera observations (false for object state)
  use_object_obs: false             # Use object state observations
  camera_height: 84                 # Camera image height (if use_camera_obs=true)
  camera_width: 84                  # Camera image width (if use_camera_obs=true)
  has_renderer: false               # Enable onscreen rendering
  has_offscreen_renderer: true      # Enable offscreen rendering (needed for camera obs/video)
  reward_shaping: true              # Use shaped rewards

alg:
  alg_name: "PPO"                   # Algorithm: "PPO", "SAC", or "TD3"
  policy: "CnnPolicy"               # "CnnPolicy" for camera, "MlpPolicy" for object state
  learning_rate: 0.0003             # Learning rate
  n_steps: 512                      # Steps per rollout
  batch_size: 64                    # Batch size (-1 for auto: PPO=64, SAC/TD3=256)
  n_epochs: 10                      # Training epochs per update
  gamma: 0.99                       # Discount factor
  gae_lambda: 0.95                  # GAE lambda
  clip_range: 0.2                   # PPO clip range
  ent_coef: 0.001                   # Entropy coefficient
  use_sde: true                     # State Dependent Exploration
  sde_sample_freq: 8                # SDE sample frequency

train:
  exp_name: ""                      # Experiment name (optional)
  n_envs: 8                         # Number of parallel environments
  device: "cuda"                    # Device: "cuda" or "cpu"
  total_timesteps: 500000           # Total training timesteps
  save_freq: 10000                  # Save checkpoint every N steps
  eval_freq: 5000                   # Evaluate every N steps
  result_save_path: "./results/"    # Path to save results
  log_path: "./logs/"               # Path for logs
  tensorboard_log: "./tensorboard_logs/"  # TensorBoard log directory

test:
  model_path: ""                    # Path to trained model (e.g., "./models/best_model.zip")
  n_episodes: 5                     # Number of test episodes
  deterministic: true               # Use deterministic actions
  save_video: true                  # Save video recordings
  video_height: 512                 # Video height
  video_width: 512                  # Video width
  result_save_path: "./results/"    # Path to save videos/results
  video_fps: 20                     # Video FPS (should match control_freq)
  device: "cuda"                    # Device: "cuda" or "cpu"
```

## Observation Types

### Camera Observations (Image-based)
- Uses RGB images from camera
- Requires `use_camera_obs: true` and `hashas_offscreen_renderer: true` in config
- Uses `CnnPolicy` to process visual input
- Automatically transposes images for PyTorch CNNs

### Object State Observations (Proprioceptive)
- Uses robot joint angles, positions, object positions, etc.
- Requires `use_object_obs: true` and `use_camera_obs: false` in config
- Uses `MlpPolicy` to process state vectors
- Can still record videos during testing

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

### Monitor Training

```bash
# View training metrics in TensorBoard
tensorboard --logdir ./tensorboard_logs/
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
└── tensorboard_logs/     # TensorBoard logs
```

## Citation

This project uses:
- [Robosuite](https://robosuite.ai/) - Robot simulation environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [MuJoCo](https://mujoco.org/) - Physics engine
