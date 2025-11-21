# ArmRL - Complete Training Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration System](#configuration-system)
3. [Algorithm Selection](#algorithm-selection)
4. [Configuration Options](#configuration-options)
5. [Training Presets](#training-presets)
6. [Examples](#examples)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring](#monitoring)

---

## Quick Start

### Train with a Config File
```bash
python train.py --config configs/fast.yaml
```

### Use Different Presets
```bash
python train.py --config configs/fast.yaml           # Quick experiments
python train.py --config configs/high_quality.yaml   # Best results
python train.py --config configs/cpu_only.yaml       # CPU-only
```

### Test Config Loading
```bash
python config.py configs/fast.yaml
```

---

## Configuration System

Training is configured via YAML files. Each config file has three sections:

### YAML Structure
```yaml
env:
  env_name: "PickPlaceCan"
  robots: ["Panda"]
  horizon: 200
  control_freq: 20
  camera_height: 84
  camera_width: 84
  use_camera_obs: true
  use_object_obs: false

alg:
  alg_name: "PPO"
  policy: "CnnPolicy"
  learning_rate: 0.0003
  n_steps: 512
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.001

train:
  n_envs: 8
  device: "cuda"
  total_timesteps: 500000
  save_freq: 10000
  eval_freq: 5000
  model_save_path: "./models/"
  log_path: "./logs/"
  tensorboard_log: "./tensorboard_logs/"
```

### Creating Custom Configs

Copy an existing config and modify as needed:
```bash
cp configs/fast.yaml configs/my_config.yaml
# Edit my_config.yaml with your settings
python train.py --config configs/my_config.yaml
```

---

## Algorithm Selection

### PPO (Proximal Policy Optimization) - Default
**Best for:** General purpose, beginners, parallel training

**Pros:**
- Stable and reliable
- Works well with 8-32 parallel environments
- Proven track record

**Cons:**
- Less sample efficient than SAC/TD3

**Config Example:**
```yaml
alg:
  alg_name: "PPO"
  policy: "CnnPolicy"
  learning_rate: 0.0003
  n_steps: 512
  batch_size: 64
```

### SAC (Soft Actor-Critic)
**Best for:** Sample efficiency, continuous control

**Pros:**
- Very sample efficient (learns faster)
- Excellent for continuous control
- Automatic entropy tuning
- Uses replay buffer (reuses past data)

**Cons:**
- Uses more memory (replay buffer)
- Benefits less from many parallel envs (use 4-8)

**Config Example:**
```yaml
alg:
  alg_name: "SAC"
  policy: "CnnPolicy"
  learning_rate: 0.0003
  batch_size: 64
  gamma: 0.99

train:
  n_envs: 4
```

### TD3 (Twin Delayed DDPG)
**Best for:** Deterministic control, sample efficiency

**Pros:**
- Very sample efficient
- Deterministic policy
- Robust performance

**Cons:**
- Uses more memory (replay buffer)
- May take longer to start learning
- Benefits less from many parallel envs (use 4-8)

**Config Example:**
```yaml
alg:
  alg_name: "TD3"
  policy: "CnnPolicy"
  learning_rate: 0.0003
  batch_size: 64

train:
  n_envs: 4
```

### Comparison Table

| Algorithm | Type | Sample Eff. | Parallel Envs | Memory | Stability |
|-----------|------|-------------|---------------|--------|-----------|
| PPO | On-Policy | Medium | High (8-32) | Low | High |
| SAC | Off-Policy | High | Low (4-8) | High | High |
| TD3 | Off-Policy | High | Low (4-8) | High | Medium |

---

## Configuration Options

### Environment Settings (`env:`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `env_name` | Robosuite environment | "PickPlaceCan" |
| `robots` | Robot type | ["Panda"] |
| `horizon` | Episode length | 200 |
| `control_freq` | Control frequency (Hz) | 20 |
| `camera_height` | Image height | 84 |
| `camera_width` | Image width | 84 |
| `use_camera_obs` | Use camera observations | true |
| `use_object_obs` | Use object state observations | false |

### Algorithm Settings (`alg:`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `alg_name` | Algorithm (PPO, SAC, TD3) | "PPO" |
| `policy` | Policy network | "CnnPolicy" |
| `learning_rate` | Learning rate | 0.0003 |
| `n_steps` | Steps per env per update (PPO) | 512 |
| `batch_size` | Batch size | 64 |
| `n_epochs` | Epochs per update (PPO) | 10 |
| `gamma` | Discount factor | 0.99 |
| `gae_lambda` | GAE lambda (PPO) | 0.95 |
| `clip_range` | PPO clip range | 0.2 |
| `ent_coef` | Entropy coefficient | 0.001 |

### Training Settings (`train:`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_envs` | Parallel environments | 8 |
| `device` | Computing device | "cuda" |
| `total_timesteps` | Total training steps | 500000 |
| `save_freq` | Checkpoint frequency | 10000 |
| `eval_freq` | Evaluation frequency | 5000 |
| `model_save_path` | Model directory | "./models/" |
| `log_path` | Log directory | "./logs/" |
| `tensorboard_log` | TensorBoard directory | "./tensorboard_logs/" |

### SAC/TD3 Hardcoded Settings
These are set in the code:
- `buffer_size: 100000` - Replay buffer size
- `tau: 0.005` - Soft update coefficient
- SAC: `ent_coef='auto'` - Automatic entropy tuning
- TD3: Action noise for exploration

---

## Training Presets

### Fast (`configs/fast.yaml`)
```bash
python train.py --config configs/fast.yaml
```
- 16 parallel environments
- 64x64 images
- 100 step episodes
- 500K timesteps
- ~2-3x faster than default

### High Quality (`configs/high_quality.yaml`)
```bash
python train.py --config configs/high_quality.yaml
```
- 8 parallel environments
- 128x128 images
- 200 step episodes
- 1M timesteps
- Better visual quality

### CPU Only (`configs/cpu_only.yaml`)
```bash
python train.py --config configs/cpu_only.yaml
```
- 8 parallel environments
- 64x64 images
- CPU device
- Smaller batch size (32)

---

## Examples

### Beginner - Start Here
```bash
python train.py --config configs/fast.yaml
```

### Compare Algorithms

Create separate configs for each algorithm:

**configs/ppo_experiment.yaml:**
```yaml
alg:
  alg_name: "PPO"
train:
  n_envs: 16
  total_timesteps: 500000
```

**configs/sac_experiment.yaml:**
```yaml
alg:
  alg_name: "SAC"
train:
  n_envs: 4
  total_timesteps: 500000
```

Then run:
```bash
python train.py --config configs/ppo_experiment.yaml
python train.py --config configs/sac_experiment.yaml

# Compare in TensorBoard
tensorboard --logdir ./tensorboard_logs/
```

### Custom High-Speed Config

Create `configs/max_speed.yaml`:
```yaml
env:
  env_name: "PickPlaceCan"
  robots: ["Panda"]
  horizon: 100
  camera_height: 64
  camera_width: 64

alg:
  alg_name: "PPO"
  n_steps: 256

train:
  n_envs: 32
  device: "cuda"
  total_timesteps: 500000
  eval_freq: 10000
  save_freq: 20000
```

```bash
python train.py --config configs/max_speed.yaml
```

---

## Performance Optimization

### Speed Optimization Tips

#### 1. Parallel Environments (Most Effective)
- **PPO:** Use 16-32 environments (scales well)
- **SAC/TD3:** Use 4-8 environments (already sample efficient)

#### 2. Image Size Reduction
- 64x64: ~6-8x faster than 256x256
- 84x84: ~3-4x faster than 256x256 (default)
- 128x128: Better quality, slower

#### 3. GPU Acceleration
- Set `device: "cuda"` in config
- Falls back to CPU if CUDA unavailable

#### 4. Episode Length
- Shorter episodes = faster iteration
- Reduce horizon to 100-150 for quicker experiments

#### 5. Evaluation Frequency
- Less frequent evaluation = faster training
- Increase eval_freq to 10000-20000

### Expected Speedups
- **Baseline** (1 env, 256x256, CPU): 1x
- **8 parallel envs**: ~6-8x
- **16 parallel envs**: ~12-15x
- **32 parallel envs**: ~20-25x
- **64x64 images**: ~6-8x
- **GPU acceleration**: ~2-3x
- **Combined** (32 envs, 64x64, GPU): **~40-60x faster!**

### Memory Considerations
- 1 env @ 84x84: ~200MB RAM
- 8 envs @ 84x84: ~1.6GB RAM
- 16 envs @ 84x84: ~3.2GB RAM
- 32 envs @ 64x64: ~2.4GB RAM
- SAC/TD3 replay buffer: +~2GB RAM

---

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ./tensorboard_logs/
# Open http://localhost:6006
```

### Check Saved Models
```bash
ls -lh ./models/
# PPO_pickplace_10000_steps.zip
# SAC_pickplace_10000_steps.zip
# TD3_pickplace_final.zip
```

### View Logs
```bash
ls ./logs/
```

### Test Trained Model
After training completes, a test video is automatically saved:
- `PPO_trained_episode.mp4`
- `SAC_trained_episode.mp4`
- `TD3_trained_episode.mp4`

---

## Recommendations by Use Case

### For Beginners
```bash
python train.py --config configs/fast.yaml
```

### For Sample Efficiency (Limited Compute)
Create a SAC config with fewer envs and more timesteps.

### For Deterministic Control
Use TD3 with `alg_name: "TD3"` in your config.

### For Maximum Speed
Use `configs/fast.yaml` or create a custom config with:
- 32 parallel envs
- 64x64 images
- horizon: 100

### For Best Results
```bash
python train.py --config configs/high_quality.yaml
```

---

## Troubleshooting

### Out of Memory
- Reduce `n_envs` (try 4 or 8)
- Reduce `camera_height` and `camera_width` to 64
- Use `device: "cpu"` if GPU memory is limited

### Training Too Slow
- Increase `n_envs` (PPO: try 16-32, SAC/TD3: try 4-8)
- Reduce image size to 64x64
- Reduce `horizon` to 100
- Use `configs/fast.yaml`

### Poor Performance
- Train longer: increase `total_timesteps` to 2000000
- Try different algorithm (SAC is often good for continuous control)
- Increase image resolution: set `camera_height: 128`
- Tune learning rate: try `learning_rate: 0.0001`

---

## Quick Reference

### Common Commands
```bash
# Fast experiment
python train.py --config configs/fast.yaml

# High quality training
python train.py --config configs/high_quality.yaml

# CPU only
python train.py --config configs/cpu_only.yaml

# Test config loading
python config.py configs/fast.yaml

# Monitor training
tensorboard --logdir ./tensorboard_logs/
```

### Algorithm Quick Picks
- **Beginner?** → Use PPO (`alg_name: "PPO"`)
- **Sample efficient?** → Use SAC (`alg_name: "SAC"`)
- **Deterministic?** → Use TD3 (`alg_name: "TD3"`)
- **Many CPU cores?** → Use PPO with high `n_envs`
- **Limited compute?** → Use SAC with fewer envs

---

## Files in This Repository

- `train.py` - Main training script
- `test_model.py` - Test trained models and save videos
- `config.py` - Configuration classes and YAML loader
- `configs/` - YAML configuration presets
  - `fast.yaml` - Quick experiments
  - `high_quality.yaml` - Best results
  - `cpu_only.yaml` - CPU-only training
- `GUIDE.md` - This comprehensive guide
- `README.md` - Quick overview and installation

That's it! You're ready to train RL agents. Start with:
```bash
python train.py --config configs/fast.yaml
```
