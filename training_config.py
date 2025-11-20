"""
Configuration options for faster PPO training with images

Usage:
    from training_config import TrainingConfig
    config = TrainingConfig()
    # Or with custom values:
    config = TrainingConfig(n_envs=16, camera_height=64)
"""


class TrainingConfig:
    """Training configuration with sensible defaults"""

    def __init__(
        self,
        # Environment settings
        env_name="PickPlaceCan",
        robots=["Panda"],
        horizon=200,
        control_freq=20,

        # Parallel environments (MOST EFFECTIVE for speedup)
        n_envs=8,  # Adjust based on CPU cores (8-32 recommended)

        # Image settings (smaller = faster)
        camera_height=84,  # 84 is standard, try 64 for more speed
        camera_width=84,

        # PPO hyperparameters
        learning_rate=3e-4,
        n_steps=512,  # Will be auto-adjusted based on n_envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        # Device
        device="cuda",  # "cuda" for GPU, "cpu" for CPU

        # Training
        total_timesteps=500000,

        # Callbacks
        save_freq=10000,
        eval_freq=5000,

        # Paths
        model_save_path="./models/",
        log_path="./logs/",
        tensorboard_log="./tensorboard_logs/",
    ):
        # Environment
        self.env_name = env_name
        self.robots = robots
        self.horizon = horizon
        self.control_freq = control_freq

        # Parallel environments
        self.n_envs = n_envs

        # Image settings
        self.camera_height = camera_height
        self.camera_width = camera_width

        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef

        # Device
        self.device = device

        # Training
        self.total_timesteps = total_timesteps

        # Callbacks
        self.save_freq = save_freq
        self.eval_freq = eval_freq

        # Paths
        self.model_save_path = model_save_path
        self.log_path = log_path
        self.tensorboard_log = tensorboard_log

    def __repr__(self):
        """Print configuration summary"""
        return f"""
Training Configuration:
=======================
Environment: {self.env_name}
Parallel Envs: {self.n_envs}
Image Size: {self.camera_height}x{self.camera_width}
Horizon: {self.horizon}
Device: {self.device}

PPO Hyperparameters:
  Learning Rate: {self.learning_rate}
  Steps per Update: {self.n_steps} x {self.n_envs} = {self.n_steps * self.n_envs}
  Batch Size: {self.batch_size}
  Epochs: {self.n_epochs}

Training:
  Total Timesteps: {self.total_timesteps:,}
  Save Frequency: {self.save_freq:,}
  Eval Frequency: {self.eval_freq:,}
"""


# Pre-defined configurations for different scenarios

class FastTrainingConfig(TrainingConfig):
    """Optimized for maximum training speed"""
    def __init__(self):
        super().__init__(
            n_envs=16,           # More parallel environments
            camera_height=64,    # Smaller images
            camera_width=64,
            horizon=100,         # Shorter episodes
            n_steps=256,         # Smaller steps (256 * 16 = 4096)
            eval_freq=10000,     # Less frequent eval
            save_freq=20000,     # Less frequent saves
        )


class HighQualityConfig(TrainingConfig):
    """Optimized for better image quality and learning"""
    def __init__(self):
        super().__init__(
            n_envs=8,
            camera_height=128,   # Higher resolution
            camera_width=128,
            horizon=200,
            total_timesteps=1000000,  # More training
        )


class CPUOnlyConfig(TrainingConfig):
    """Optimized for CPU-only training"""
    def __init__(self):
        super().__init__(
            n_envs=8,
            camera_height=64,
            camera_width=64,
            device="cpu",
            batch_size=32,       # Smaller batches for CPU
        )


# ====================
# OPTIMIZATION NOTES
# ====================
"""
Memory Usage:
-------------
- Main thread: MEM~8GB, GPU~3GB
- Each env instance: MEM~4GB, GPU~1GB
"""