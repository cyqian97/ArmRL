# Changelog

## Command-Line Interface Support

### Summary
Added comprehensive command-line argument support to `test_env.py`, allowing you to configure all training parameters from the command line without editing code.

### What Changed

#### 1. **test_env.py** - Main Changes
- Added `argparse` for command-line argument parsing
- Added `parse_args()` function to handle all configuration parameters
- Support for 4 preset configurations: `default`, `fast`, `high_quality`, `cpu`
- All training parameters can now be specified via command line
- Configuration is printed at startup for verification

#### 2. **New Files Created**

**Configuration & Documentation:**
- `training_config.py` - Configuration classes (TrainingConfig, FastTrainingConfig, etc.)
- `USAGE.md` - Detailed usage guide with all parameters
- `QUICK_START.md` - Quick reference with common examples
- `compare_configs.sh` - Script to compare different configurations
- `show_help.py` - Display help without loading dependencies
- `example_configs.py` - View and compare all preset configurations

**This file:**
- `CHANGELOG.md` - Summary of changes

### How to Use

#### Basic Usage
```bash
# Default configuration
python test_env.py

# Use a preset
python test_env.py --preset fast
python test_env.py --preset high_quality
python test_env.py --preset cpu

# Custom parameters
python test_env.py --n_envs 16 --camera_height 64
python test_env.py --total_timesteps 1000000
```

#### View All Options
```bash
python test_env.py --help
python show_help.py  # If dependencies not installed
```

#### Compare Configurations
```bash
./compare_configs.sh
```

### Available Command-Line Arguments

#### Presets
- `--preset` - Choose from: default, fast, high_quality, cpu

#### Environment Settings
- `--env_name` - Robosuite environment (default: PickPlaceCan)
- `--horizon` - Episode length (default: 200)

#### Parallelization
- `--n_envs` - Number of parallel environments (default: 8)

#### Image Settings
- `--camera_height` - Image height (default: 84)
- `--camera_width` - Image width (default: 84)

#### PPO Hyperparameters
- `--learning_rate` - Learning rate (default: 3e-4)
- `--n_steps` - Steps per env per update (default: 512)
- `--batch_size` - Batch size (default: 64)
- `--n_epochs` - Epochs per update (default: 10)
- `--gamma` - Discount factor (default: 0.99)
- `--gae_lambda` - GAE lambda (default: 0.95)
- `--clip_range` - PPO clip range (default: 0.2)
- `--ent_coef` - Entropy coefficient (default: 0.01)

#### Device
- `--device` - cuda or cpu (default: cuda)

#### Training
- `--total_timesteps` - Total training steps (default: 500000)

#### Callbacks
- `--save_freq` - Checkpoint save frequency (default: 10000)
- `--eval_freq` - Evaluation frequency (default: 5000)

#### Paths
- `--model_save_path` - Model directory (default: ./models/)
- `--log_path` - Log directory (default: ./logs/)
- `--tensorboard_log` - TensorBoard directory (default: ./tensorboard_logs/)

### Examples

#### Quick Experiments
```bash
python test_env.py --preset fast
# 16 envs, 64x64 images, 100 step episodes
# ~2-3x faster than default
```

#### Maximum Speed (with 64 CPU cores)
```bash
python test_env.py --n_envs 32 --camera_height 64 --horizon 100 --n_steps 256
# ~4-5x faster than default
```

#### Long Training Run
```bash
python test_env.py --preset high_quality
# 128x128 images, 1M timesteps
```

#### CPU-Only Training
```bash
python test_env.py --preset cpu
# Or: python test_env.py --device cpu --batch_size 32
```

#### Custom Hyperparameter Tuning
```bash
python test_env.py \
  --n_envs 16 \
  --learning_rate 0.0001 \
  --batch_size 128 \
  --n_epochs 20 \
  --total_timesteps 2000000
```

### Backward Compatibility

The old method of editing the code still works if you prefer:
```python
# In test_env.py, lines 102-113
config = TrainingConfig(n_envs=16, camera_height=64)
```

However, command-line arguments are now the recommended approach as they:
- Don't require code edits
- Are easier to script and automate
- Can be version controlled separately
- Allow easy experimentation

### Performance Tips

1. **Increase parallelization**: Use `--n_envs 16` or higher
2. **Reduce image size**: Use `--camera_height 64 --camera_width 64`
3. **Shorter episodes**: Use `--horizon 100`
4. **Less frequent eval**: Use `--eval_freq 10000`

### Monitoring

```bash
# TensorBoard
tensorboard --logdir ./tensorboard_logs/

# Check models
ls -lh ./models/

# View configuration comparison
./compare_configs.sh
```
