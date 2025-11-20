# Quick Start Guide

## Command Line Training Examples

### 1. Default Training
```bash
python test_env.py
```

### 2. Preset Configurations

#### Fast Training (Quick Experiments)
```bash
python test_env.py --preset fast
```
- 16 parallel environments
- 64×64 images
- Shorter episodes (100 steps)
- Less frequent evaluation

#### High Quality Training (Final Run)
```bash
python test_env.py --preset high_quality
```
- 8 parallel environments
- 128×128 images (higher resolution)
- Full episodes (200 steps)
- Train for 1M timesteps

#### CPU-Only Training
```bash
python test_env.py --preset cpu
```
- Optimized for CPU
- 8 environments
- 64×64 images
- Smaller batch size

### 3. Common Custom Configurations

#### More Parallel Environments
```bash
python test_env.py --n_envs 16
```

#### Smaller Images (Faster)
```bash
python test_env.py --camera_height 64 --camera_width 64
```

#### Train Longer
```bash
python test_env.py --total_timesteps 1000000
```

#### Combine Multiple Settings
```bash
python test_env.py --n_envs 16 --camera_height 64 --total_timesteps 1000000
```

### 4. Maximum Speed Configuration (Your 64 CPU Cores)
```bash
python test_env.py \
  --n_envs 32 \
  --camera_height 64 \
  --camera_width 64 \
  --horizon 100 \
  --n_steps 256
```

### 5. Tune PPO Hyperparameters
```bash
python test_env.py \
  --learning_rate 0.0001 \
  --n_steps 512 \
  --batch_size 128 \
  --n_epochs 20
```

## Key Parameters Explained

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--n_envs` | Parallel environments | 8 | 8-32 |
| `--camera_height` | Image height | 84 | 64-128 |
| `--camera_width` | Image width | 84 | 64-128 |
| `--horizon` | Episode length | 200 | 100-300 |
| `--n_steps` | Steps per update | 512 | 256-2048 |
| `--total_timesteps` | Total training steps | 500,000 | 500K-2M |
| `--device` | cuda or cpu | cuda | - |

## View All Options
```bash
python test_env.py --help
```

## Monitor Training

### TensorBoard
```bash
tensorboard --logdir ./tensorboard_logs/
```
Then open http://localhost:6006 in your browser

### Check Saved Models
```bash
ls -lh ./models/
```

## Speedup Tips

1. **Increase parallel envs**: `--n_envs 16` or `--n_envs 32`
2. **Reduce image size**: `--camera_height 64 --camera_width 64`
3. **Shorter episodes**: `--horizon 100`
4. **Use GPU**: `--device cuda` (automatic if available)

## Files Created

- `training_config.py` - Configuration classes
- `test_env.py` - Main training script with CLI support
- `USAGE.md` - Detailed usage guide
- `QUICK_START.md` - This file
- `example_configs.py` - View all preset configs
