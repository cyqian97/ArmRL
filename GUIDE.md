# ArmRL - Complete Training Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Algorithm Selection](#algorithm-selection)
3. [Command Line Arguments](#command-line-arguments)
4. [Training Presets](#training-presets)
5. [Examples](#examples)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring](#monitoring)

---

## Quick Start

### Default Training (PPO)
```bash
python test_env.py
```

### Try Different Algorithms
```bash
python test_env.py --algo ppo    # Stable, parallel-friendly
python test_env.py --algo sac    # Sample efficient
python test_env.py --algo td3    # Deterministic
```

### Use Presets
```bash
python test_env.py --preset fast           # Quick experiments
python test_env.py --preset high_quality   # Best results
python test_env.py --preset cpu            # CPU-only
```

### View All Options
```bash
python test_env.py --help
```

---

## Algorithm Selection

### PPO (Proximal Policy Optimization) - Default
**Best for:** General purpose, beginners, parallel training

**Pros:**
- ✅ Stable and reliable
- ✅ Works well with 8-32 parallel environments
- ✅ Proven track record

**Cons:**
- ❌ Less sample efficient than SAC/TD3

**Example:**
```bash
python test_env.py --algo ppo --n_envs 16 --total_timesteps 500000
```

### SAC (Soft Actor-Critic)
**Best for:** Sample efficiency, continuous control

**Pros:**
- ✅ Very sample efficient (learns faster)
- ✅ Excellent for continuous control
- ✅ Automatic entropy tuning
- ✅ Uses replay buffer (reuses past data)

**Cons:**
- ❌ Uses more memory (replay buffer)
- ❌ Benefits less from many parallel envs (use 4-8)

**Example:**
```bash
python test_env.py --algo sac --n_envs 4 --total_timesteps 1000000
```

### TD3 (Twin Delayed DDPG)
**Best for:** Deterministic control, sample efficiency

**Pros:**
- ✅ Very sample efficient
- ✅ Deterministic policy
- ✅ Robust performance

**Cons:**
- ❌ Uses more memory (replay buffer)
- ❌ May take longer to start learning
- ❌ Benefits less from many parallel envs (use 4-8)

**Example:**
```bash
python test_env.py --algo td3 --n_envs 4 --total_timesteps 1000000
```

### Comparison Table

| Algorithm | Type | Sample Eff. | Parallel Envs | Memory | Stability |
|-----------|------|-------------|---------------|--------|-----------|
| PPO | On-Policy | Medium ⭐⭐ | High (8-32) | Low | High ⭐⭐⭐ |
| SAC | Off-Policy | High ⭐⭐⭐ | Low (4-8) | High | High ⭐⭐⭐ |
| TD3 | Off-Policy | High ⭐⭐⭐ | Low (4-8) | High | Medium ⭐⭐ |

---

## Command Line Arguments

### Algorithm & Presets
- `--algo {ppo,sac,td3}` - RL algorithm (default: ppo)
- `--preset {default,fast,high_quality,cpu}` - Preset configuration

### Environment Settings
- `--env_name` - Robosuite environment (default: PickPlaceCan)
- `--horizon` - Episode length (default: 200)
- `--n_envs` - Parallel environments (default: 8)

### Image Settings
- `--camera_height` - Image height (default: 84)
- `--camera_width` - Image width (default: 84)

### PPO Hyperparameters (PPO only)
- `--learning_rate` - Learning rate (default: 3e-4)
- `--n_steps` - Steps per env per update (default: 512)
- `--batch_size` - Batch size (default: 64)
- `--n_epochs` - Epochs per update (default: 10)
- `--gamma` - Discount factor (default: 0.99)
- `--gae_lambda` - GAE lambda (default: 0.95)
- `--clip_range` - PPO clip range (default: 0.2)
- `--ent_coef` - Entropy coefficient (default: 0.01)

### SAC/TD3 Settings (hardcoded in code)
- `buffer_size: 100000` - Replay buffer size
- `tau: 0.005` - Soft update coefficient
- SAC: `ent_coef='auto'` - Automatic entropy tuning
- TD3: Action noise for exploration

### Training
- `--device {cuda,cpu}` - Computing device (default: cuda)
- `--total_timesteps` - Total training steps (default: 500000)
- `--save_freq` - Checkpoint frequency (default: 10000)
- `--eval_freq` - Evaluation frequency (default: 5000)

### Paths
- `--model_save_path` - Model directory (default: ./models/)
- `--log_path` - Log directory (default: ./logs/)
- `--tensorboard_log` - TensorBoard directory (default: ./tensorboard_logs/)

---

## Training Presets

### Fast (Quick Experiments)
```bash
python test_env.py --preset fast
```
- 16 parallel environments
- 64×64 images
- 100 step episodes
- 500K timesteps
- ~2-3x faster than default

### High Quality (Final Training)
```bash
python test_env.py --preset high_quality
```
- 8 parallel environments
- 128×128 images
- 200 step episodes
- 1M timesteps
- Better visual quality

### CPU Only
```bash
python test_env.py --preset cpu
```
- 8 parallel environments
- 64×64 images
- CPU device
- Smaller batch size (32)

---

## Examples

### Beginner - Start Here
```bash
python test_env.py
```

### Compare Algorithms
```bash
# Train all three
python test_env.py --algo ppo --n_envs 16 --total_timesteps 500000
python test_env.py --algo sac --n_envs 4 --total_timesteps 500000
python test_env.py --algo td3 --n_envs 4 --total_timesteps 500000

# Compare in TensorBoard
tensorboard --logdir ./tensorboard_logs/
```

### Algorithm + Preset
```bash
python test_env.py --algo sac --preset fast
python test_env.py --algo td3 --preset high_quality
```

### Custom Configurations
```bash
# More parallel environments (PPO)
python test_env.py --algo ppo --n_envs 32

# Sample efficient training (SAC)
python test_env.py --algo sac --n_envs 4 --total_timesteps 2000000

# Smaller images (faster)
python test_env.py --camera_height 64 --camera_width 64

# Different environment
python test_env.py --env_name Stack

# Custom hyperparameters
python test_env.py --algo ppo --learning_rate 0.0001 --batch_size 128
```

### Maximum Speed (64 CPU cores)
```bash
# PPO - maximize parallelization
python test_env.py \
  --algo ppo \
  --n_envs 32 \
  --camera_height 64 \
  --camera_width 64 \
  --horizon 100 \
  --n_steps 256

# SAC - balance efficiency and speed
python test_env.py \
  --algo sac \
  --n_envs 8 \
  --camera_height 64 \
  --total_timesteps 2000000
```

---

## Performance Optimization

### Speed Optimization Tips

#### 1. Parallel Environments (Most Effective)
- **PPO:** Use 16-32 environments (scales well)
- **SAC/TD3:** Use 4-8 environments (already sample efficient)

```bash
# PPO with maximum parallelization
python test_env.py --algo ppo --n_envs 32
```

#### 2. Image Size Reduction
- 64×64: ~6-8x faster than 256×256
- 84×84: ~3-4x faster than 256×256 (default)
- 128×128: Better quality, slower

```bash
python test_env.py --camera_height 64 --camera_width 64
```

#### 3. GPU Acceleration
- Automatically uses CUDA if available
- Can force CPU with `--device cpu`

```bash
python test_env.py --device cuda
```

#### 4. Episode Length
- Shorter episodes = faster iteration
- Reduce horizon to 100-150 for quicker experiments

```bash
python test_env.py --horizon 100
```

#### 5. Evaluation Frequency
- Less frequent evaluation = faster training
- Increase eval_freq to 10000-20000

```bash
python test_env.py --eval_freq 10000
```

### Expected Speedups
- **Baseline** (1 env, 256×256, CPU): 1x
- **8 parallel envs**: ~6-8x
- **16 parallel envs**: ~12-15x
- **32 parallel envs**: ~20-25x
- **64×64 images**: ~6-8x
- **GPU acceleration**: ~2-3x
- **Combined** (32 envs, 64×64, GPU): **~40-60x faster!**

### Memory Considerations
- 1 env @ 84×84: ~200MB RAM
- 8 envs @ 84×84: ~1.6GB RAM
- 16 envs @ 84×84: ~3.2GB RAM
- 32 envs @ 64×64: ~2.4GB RAM
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
# ppo_pickplace_10000_steps.zip
# sac_pickplace_10000_steps.zip
# td3_pickplace_final.zip
```

### View Logs
```bash
ls ./logs/
```

### Test Trained Model
After training completes, a test video is automatically saved:
- `ppo_trained_episode.mp4`
- `sac_trained_episode.mp4`
- `td3_trained_episode.mp4`

---

## Recommendations by Use Case

### For Beginners
```bash
python test_env.py --preset fast
```

### For Sample Efficiency (Limited Compute)
```bash
python test_env.py --algo sac --n_envs 4 --total_timesteps 1000000
```

### For Deterministic Control
```bash
python test_env.py --algo td3 --n_envs 4 --total_timesteps 1000000
```

### For Maximum Speed (64 CPU Cores)
```bash
python test_env.py --algo ppo --n_envs 32 --camera_height 64 --horizon 100
```

### For Best Results (Patient Training)
```bash
python test_env.py --preset high_quality
```

---

## Troubleshooting

### Out of Memory
- Reduce `--n_envs` (try 4 or 8)
- Reduce `--camera_height` and `--camera_width` to 64
- Use `--device cpu` if GPU memory is limited

### Training Too Slow
- Increase `--n_envs` (PPO: try 16-32, SAC/TD3: try 4-8)
- Reduce image size to 64×64
- Reduce `--horizon` to 100
- Use `--preset fast`

### Poor Performance
- Train longer: `--total_timesteps 2000000`
- Try different algorithm (SAC is often good for continuous control)
- Increase image resolution: `--camera_height 128`
- Tune learning rate: `--learning_rate 0.0001`

---

## Quick Reference

### Common Commands
```bash
# Default
python test_env.py

# Fast experiment
python test_env.py --preset fast

# SAC sample efficient
python test_env.py --algo sac --n_envs 4 --total_timesteps 1000000

# PPO maximum speed
python test_env.py --algo ppo --n_envs 32 --camera_height 64

# View help
python test_env.py --help

# Monitor training
tensorboard --logdir ./tensorboard_logs/
```

### Algorithm Quick Picks
- **Beginner?** → Use PPO (`--algo ppo`)
- **Sample efficient?** → Use SAC (`--algo sac`)
- **Deterministic?** → Use TD3 (`--algo td3`)
- **Many CPU cores?** → Use PPO with high `--n_envs`
- **Limited compute?** → Use SAC with fewer envs

---

## Files in This Repository

- `test_env.py` - Main training script
- `training_config.py` - Configuration classes
- `GUIDE.md` - This comprehensive guide
- `README.md` - Quick overview and installation

That's it! You're ready to train RL agents. Start with:
```bash
python test_env.py --preset fast
```
