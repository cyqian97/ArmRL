"""
Examples of how to use training_config.py

Just uncomment the configuration you want to use in test_env.py
"""

from training_config import TrainingConfig, FastTrainingConfig, HighQualityConfig

# ============================================
# EXAMPLE 1: Default Configuration
# ============================================
# Balanced settings for most use cases
print("=" * 50)
print("EXAMPLE 1: Default Configuration")
print("=" * 50)
config1 = TrainingConfig()
print(config1)


# ============================================
# EXAMPLE 2: Fast Training (Quick Experiments)
# ============================================
# Use this when you want to iterate quickly
print("\n" + "=" * 50)
print("EXAMPLE 2: Fast Training Config")
print("=" * 50)
config2 = FastTrainingConfig()
print(config2)


# ============================================
# EXAMPLE 3: High Quality (Final Training)
# ============================================
# Use this for final training runs with better quality
print("\n" + "=" * 50)
print("EXAMPLE 3: High Quality Config")
print("=" * 50)
config3 = HighQualityConfig()
print(config3)


# ============================================
# EXAMPLE 4: Custom Configuration
# ============================================
# Create your own custom settings
print("\n" + "=" * 50)
print("EXAMPLE 4: Custom Configuration")
print("=" * 50)
config4 = TrainingConfig(
    n_envs=16,              # 16 parallel environments
    camera_height=64,       # Smaller images for speed
    camera_width=64,
    horizon=150,            # Medium episode length
    total_timesteps=1000000,  # Train for 1M steps
    learning_rate=1e-4,     # Lower learning rate
    device="cuda"           # Use GPU
)
print(config4)


# ============================================
# EXAMPLE 5: CPU-Only Training
# ============================================
print("\n" + "=" * 50)
print("EXAMPLE 5: CPU-Only Training")
print("=" * 50)
config5 = TrainingConfig(
    n_envs=8,
    camera_height=64,
    camera_width=64,
    device="cpu",
    batch_size=32
)
print(config5)


# ============================================
# EXAMPLE 6: Maximum Speed Configuration
# ============================================
# Maximum parallelization for fastest training
print("\n" + "=" * 50)
print("EXAMPLE 6: Maximum Speed Config")
print("=" * 50)
config6 = TrainingConfig(
    n_envs=32,              # Maximum parallel envs (adjust based on RAM)
    camera_height=64,       # Small images
    camera_width=64,
    horizon=100,            # Short episodes
    n_steps=256,            # Small steps (256 * 32 = 8192)
    eval_freq=20000,        # Less frequent evaluation
    save_freq=50000,        # Less frequent saving
)
print(config6)


print("\n" + "=" * 50)
print("To use any of these configs in test_env.py:")
print("=" * 50)
print("""
1. Open test_env.py
2. Go to line 102-113
3. Uncomment the config you want to use

Examples:
---------
# For quick experiments:
config = FastTrainingConfig()

# For custom settings:
config = TrainingConfig(n_envs=16, camera_height=64)

# For maximum speed (if you have 64 CPU cores):
config = TrainingConfig(n_envs=32, camera_height=64, horizon=100)
""")
