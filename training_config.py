"""
Configuration options for faster PPO training with images
"""

# ====================
# TRAINING ACCELERATION OPTIONS
# ====================

# 1. Parallel Environments (MOST EFFECTIVE)
# ------------------------------------------
# Use SubprocVecEnv to run multiple environments in parallel
# Speedup: ~linear with number of cores (8 envs = ~8x faster data collection)
N_ENVS = 8  # Start with 8, can go up to 16 depending on CPU cores

# 2. Image Size Reduction
# ------------------------------------------
# Smaller images = faster CNN forward/backward passes
# 84x84 is standard for Atari, 64x64 can work too
CAMERA_HEIGHT = 84
CAMERA_WIDTH = 84
# Alternative: Try 64x64 for even faster training
# CAMERA_HEIGHT = 64
# CAMERA_WIDTH = 64

# 3. GPU Acceleration
# ------------------------------------------
# Use CUDA if available (set device="cuda" in PPO)
USE_GPU = True  # Automatically falls back to CPU if no GPU

# 4. Optimized Hyperparameters for Parallel Envs
# ------------------------------------------
# With N parallel envs, reduce n_steps accordingly
# Total steps per update = n_steps * n_envs
N_STEPS = 512  # 512 * 8 = 4096 steps per update (good default)
BATCH_SIZE = 128  # Larger batches for GPU efficiency
N_EPOCHS = 10

# 5. Frame Stacking (Optional - can help learning)
# ------------------------------------------
# Stack multiple frames to give temporal information
# Increases memory but can improve learning
USE_FRAME_STACK = False  # Set to True to enable
N_STACK = 4  # Number of frames to stack

# 6. Action Repeat (Optional - even faster)
# ------------------------------------------
# Repeat each action for N steps (reduces effective horizon)
# Can speed up training but may reduce control precision
USE_ACTION_REPEAT = False
ACTION_REPEAT = 2

# 7. Mixed Precision Training (Optional - experimental)
# ------------------------------------------
# Can speed up GPU training but may affect stability
USE_MIXED_PRECISION = False

# ====================
# ESTIMATED SPEEDUP
# ====================
# Baseline (1 env, 256x256 images, CPU): 1x
# With optimizations:
# - 8 parallel envs: ~6-8x faster
# - 84x84 images: ~3-4x faster
# - GPU acceleration: ~2-3x faster
# Combined: ~20-50x faster than baseline!

# ====================
# MEMORY CONSIDERATIONS
# ====================
# Each environment with 84x84x3 images uses ~200MB
# 8 parallel envs ≈ 1.6GB RAM
# 16 parallel envs ≈ 3.2GB RAM
# Adjust N_ENVS based on available RAM

# ====================
# ADDITIONAL TIPS
# ====================
# 1. Reduce evaluation frequency (eval_freq) to save time
# 2. Reduce checkpoint frequency (save_freq) to save disk I/O
# 3. Use environment horizon=100 instead of 200 for faster episodes
# 4. Consider using SAC or TD3 which can be sample-efficient for continuous control
