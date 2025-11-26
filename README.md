# ArmRL - RL Training for Robotic Manipulation

Train RL agents (PPO, SAC, TD3) with image observations for robotic manipulation tasks using Robosuite and stable-baselines3.

**Supported Algorithms:** PPO | SAC | TD3

## Installation

```bash
apt-get update -y
apt-get install -y libglib2.0-0

conda create -n armrl python=3.12 -y
conda activate armrl
pip install mujoco robosuite "imageio[ffmpeg]" opencv-python PyYAML
pip install gymnasium stable-baselines3[extra] tensorboard h5py
```

## Quick Start

```bash
# Train with a config file
python train.py --config configs/fast.yaml

# Use different presets
python train.py --config configs/high_quality.yaml
python train.py --config configs/cpu_only.yaml

# Test config loading
python config.py configs/fast.yaml
```

## Configuration

Training is configured via YAML files in the `configs/` directory:

| Config File | Description |
|-------------|-------------|
| `configs/fast.yaml` | Quick experiments (64x64 images, 16 envs) |
| `configs/high_quality.yaml` | Best results (128x128 images, 1M steps) |
| `configs/cpu_only.yaml` | CPU-only training |


## Algorithm Comparison

| Algorithm | Best For | Parallel Envs | Sample Efficiency | Stability |
|-----------|----------|---------------|-------------------|-----------|
| **PPO** | General purpose, beginners | 8-32 | Medium | High |
| **SAC** | Sample efficiency | 4-8 | High | High |
| **TD3** | Deterministic control | 4-8 | High | Medium |

## Monitor Training
```bash
tensorboard --logdir ./tensorboard_logs/
```

## Test Trained Models

After training, test your model and save a video:

```bash
# Test the trained model (auto-detects algorithm)
python test_model.py --model ppo_pickplace_final.zip

# Run multiple episodes
python test_model.py --model sac_pickplace_final.zip --episodes 5

# Custom video name
python test_model.py --model ppo_pickplace_final.zip --video my_test.mp4

# Test checkpoint model
python test_model.py --model models/ppo_pickplace_10000_steps.zip
```

## Complete Documentation

See **[GUIDE.md](GUIDE.md)** for:
- Detailed algorithm explanations
- All configuration options
- Performance optimization tips
- Training examples
- Troubleshooting

## Files

- `train.py` - Main training script
- `test_model.py` - Test trained models and save videos
- `config.py` - Configuration classes and YAML loader
- `configs/` - YAML configuration presets
- `GUIDE.md` - Complete training guide
- `README.md` - This file