# ArmRL - RL Training for Robotic Manipulation

Train RL agents (PPO, SAC, TD3) with image observations for robotic manipulation tasks using Robosuite and stable-baselines3.

**Supported Algorithms:** PPO | SAC | TD3

## Installation

```bash
apt-get update -y
apt-get install -y libglib2.0-0

conda create -n armrl python=3.12 -y
conda activate armrl
pip install mujoco robosuite "imageio[ffmpeg]" opencv-python
pip install gymnasium stable-baselines3[extra] tensorboard
```

## Quick Start

```bash
# Default training (PPO)
python test_env.py

# Try different algorithms
python test_env.py --algo sac    # Sample efficient
python test_env.py --algo td3    # Deterministic

# Use presets
python test_env.py --preset fast           # Quick experiments
python test_env.py --preset high_quality   # Best results

# Custom configuration
python test_env.py --algo sac --n_envs 4 --total_timesteps 1000000
python test_env.py --algo ppo --n_envs 32 --camera_height 64

# View all options
python test_env.py --help
```

## Algorithm Comparison

| Algorithm | Best For | Parallel Envs | Sample Efficiency | Stability |
|-----------|----------|---------------|-------------------|-----------|
| **PPO** | General purpose, beginners | 8-32 | Medium ⭐⭐ | High ⭐⭐⭐ |
| **SAC** | Sample efficiency | 4-8 | High ⭐⭐⭐ | High ⭐⭐⭐ |
| **TD3** | Deterministic control | 4-8 | High ⭐⭐⭐ | Medium ⭐⭐ |

## Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--algo` | RL algorithm | ppo | ppo, sac, td3 |
| `--preset` | Preset config | None | fast, high_quality, cpu |
| `--n_envs` | Parallel environments | 8 | 4-32 |
| `--camera_height` | Image height | 84 | 64/84/128 |
| `--total_timesteps` | Training steps | 500,000 | Any |
| `--device` | Computing device | cuda | cuda/cpu |

## Monitor Training
```bash
tensorboard --logdir ./tensorboard_logs/
```

## Complete Documentation

See **[GUIDE.md](GUIDE.md)** for:
- Detailed algorithm explanations
- All command-line arguments
- Performance optimization tips
- Training examples
- Troubleshooting

## Recommendations

**Beginner?** → `python test_env.py --preset fast`

**Sample efficient?** → `python test_env.py --algo sac --n_envs 4 --total_timesteps 1000000`

**Maximum speed?** → `python test_env.py --algo ppo --n_envs 32 --camera_height 64`

## Files

- `test_env.py` - Main training script
- `training_config.py` - Configuration classes
- `GUIDE.md` - Complete training guide
- `README.md` - This file