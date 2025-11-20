# ArmRL - PPO Training for Robotic Manipulation

Train PPO agents with image observations for robotic manipulation tasks using Robosuite and stable-baselines3.

## Installation

```bash
apt-get update -y
apt-get install -y libglib2.0-0
```

```bash
conda create -n armrl python=3.12 -y
conda activate armrl
pip install mujoco
pip install robosuite
pip install "imageio[ffmpeg]"  # For saving videos
pip install opencv-python  # For camera observations
pip install gymnasium
pip install stable-baselines3[extra]  # For training RL agents
pip install tensorboard
```

## Quick Start

### 1. Default Training
```bash
python test_env.py
```

### 2. Use Presets
```bash
# Fast training (quick experiments)
python test_env.py --preset fast

# High quality (final training)
python test_env.py --preset high_quality

# CPU-only
python test_env.py --preset cpu
```

### 3. Custom Configuration
```bash
# 16 parallel environments with smaller images
python test_env.py --n_envs 16 --camera_height 64

# Train for 1 million steps
python test_env.py --total_timesteps 1000000

# Maximum speed (with 64 CPU cores)
python test_env.py --n_envs 32 --camera_height 64 --horizon 100
```

## View All Options
```bash
python test_env.py --help
```

## Monitor Training
```bash
tensorboard --logdir ./tensorboard_logs/
```

## Documentation

- [QUICK_START.md](QUICK_START.md) - Quick reference with examples
- [USAGE.md](USAGE.md) - Detailed usage guide with all parameters
- `./compare_configs.sh` - Compare different configurations

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--preset` | Use preset config (fast/high_quality/cpu) | None |
| `--n_envs` | Number of parallel environments | 8 |
| `--camera_height` | Image height (64/84/128) | 84 |
| `--n_steps` | Steps per env per update | 512 |
| `--total_timesteps` | Total training steps | 500,000 |
| `--device` | cuda or cpu | cuda |

## Files

- `test_env.py` - Main training script with CLI support
- `training_config.py` - Configuration classes
- `example_configs.py` - View all preset configurations
- `show_help.py` - Display help without loading dependencies