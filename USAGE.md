# PPO Training with Command Line Arguments

## Quick Start

### 1. Use Default Configuration
```bash
python test_env.py
```

### 2. Use Preset Configurations
```bash
# Fast training (16 envs, 64x64 images, shorter episodes)
python test_env.py --preset fast

# High quality training (128x128 images, longer training)
python test_env.py --preset high_quality

# CPU-only training
python test_env.py --preset cpu
```

### 3. Custom Configuration via Command Line

#### Basic Examples
```bash
# 16 parallel environments with smaller images
python test_env.py --n_envs 16 --camera_height 64 --camera_width 64

# Train for 1 million timesteps
python test_env.py --total_timesteps 1000000

# Use CPU instead of GPU
python test_env.py --device cpu

# Shorter episodes for faster training
python test_env.py --horizon 100
```

#### Advanced Examples
```bash
# Maximum speed training (with your 64 CPU cores)
python test_env.py \
  --n_envs 32 \
  --camera_height 64 \
  --camera_width 64 \
  --horizon 100 \
  --n_steps 256 \
  --total_timesteps 1000000

# Fine-tuned PPO hyperparameters
python test_env.py \
  --n_envs 16 \
  --learning_rate 0.0001 \
  --n_steps 512 \
  --batch_size 128 \
  --n_epochs 20 \
  --gamma 0.99 \
  --total_timesteps 2000000

# Different environment
python test_env.py --env_name Stack --n_envs 8
```

## All Available Arguments

### Environment Settings
- `--env_name` - Robosuite environment name (default: `PickPlaceCan`)
- `--horizon` - Episode length (default: `200`)

### Parallel Environments
- `--n_envs` - Number of parallel environments (default: `8`)

### Image Settings
- `--camera_height` - Camera image height (default: `84`)
- `--camera_width` - Camera image width (default: `84`)

### PPO Hyperparameters
- `--learning_rate` - Learning rate (default: `3e-4`)
- `--n_steps` - Steps per environment per update (default: `512`)
- `--batch_size` - Batch size (default: `64`)
- `--n_epochs` - Number of epochs (default: `10`)
- `--gamma` - Discount factor (default: `0.99`)
- `--gae_lambda` - GAE lambda (default: `0.95`)
- `--clip_range` - PPO clip range (default: `0.2`)
- `--ent_coef` - Entropy coefficient (default: `0.01`)

### Device
- `--device` - Device to use: `cuda` or `cpu` (default: `cuda`)

### Training
- `--total_timesteps` - Total training timesteps (default: `500000`)

### Callbacks
- `--save_freq` - Save checkpoint frequency (default: `10000`)
- `--eval_freq` - Evaluation frequency (default: `5000`)

### Paths
- `--model_save_path` - Model save directory (default: `./models/`)
- `--log_path` - Log directory (default: `./logs/`)
- `--tensorboard_log` - TensorBoard log directory (default: `./tensorboard_logs/`)

## View Help
```bash
python test_env.py --help
```

## Recommended Configurations

### For Quick Experiments (Fast Iteration)
```bash
python test_env.py --preset fast
# OR
python test_env.py --n_envs 16 --camera_height 64 --horizon 100 --total_timesteps 200000
```

### For Best Performance (Final Training)
```bash
python test_env.py --preset high_quality
# OR
python test_env.py --camera_height 128 --total_timesteps 2000000
```

### For Limited Resources (CPU Only)
```bash
python test_env.py --preset cpu
# OR
python test_env.py --device cpu --n_envs 4 --camera_height 64 --batch_size 32
```

### For Maximum Speed (64 CPU cores)
```bash
python test_env.py \
  --n_envs 32 \
  --camera_height 64 \
  --camera_width 64 \
  --horizon 100 \
  --n_steps 256 \
  --eval_freq 20000 \
  --save_freq 50000
```

## Monitor Training

### TensorBoard
```bash
tensorboard --logdir ./tensorboard_logs/
```

### Check Saved Models
```bash
ls -lh ./models/
```

### View Logs
```bash
cat ./logs/evaluations.npz
```

## Tips

1. **n_steps × n_envs** should be around 2048-8192 for good PPO performance
2. Start with `--preset fast` for debugging, then use `--preset high_quality` for final training
3. Increase `n_envs` based on available CPU cores (you have 64!)
4. Decrease `camera_height`/`camera_width` to 64 for faster training
5. Use `--device cpu` if no GPU is available
6. Reduce `horizon` to 100-150 for faster episode completion
