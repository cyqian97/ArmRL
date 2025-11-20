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
        camera_height=84,  # 84 is standard, try 64 for more speed
        camera_width=84,
        use_object_obs=False, 
        use_camera_obs=True,

        # PPO hyperparameters
        learning_rate=3e-4,
        n_steps=512,  # Will be auto-adjusted based on n_envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,

        # Training settings
        n_envs=8,  
        device="cuda",  # "cuda" for GPU, "cpu" for CPU
        total_timesteps=500000,
        save_freq=10000, # Callbacks frequencies
        eval_freq=5000,
        model_save_path="./models/",
        log_path="./logs/",
        tensorboard_log="./tensorboard_logs/",
    ):
        # Environment
        self.env_name = env_name
        self.robots = robots
        self.horizon = horizon
        self.control_freq = control_freq
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.use_object_obs = use_object_obs
        self.use_camera_obs = use_camera_obs

        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef

        # Training parameters
        self.device = device        
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path
        self.log_path = log_path
        self.tensorboard_log = tensorboard_log

    def __repr__(self):
        """Print configuration summary"""
        return f"""
Training Configuration:
=======================

Environment Settings:
  Environment Name: {self.env_name}
  Robots: {self.robots}
  Horizon: {self.horizon}
  Control Frequency: {self.control_freq} Hz
  Camera Height: {self.camera_height}
  Camera Width: {self.camera_width}
  Image Size: {self.camera_height}x{self.camera_width}
  Use Object Obs: {self.use_object_obs}
  Use Camera Obs: {self.use_camera_obs}

PPO Hyperparameters:
  Learning Rate: {self.learning_rate}
  N Steps: {self.n_steps}
  Steps per Update: {self.n_steps} x {self.n_envs} = {self.n_steps * self.n_envs:,}
  Batch Size: {self.batch_size}
  N Epochs: {self.n_epochs}
  Gamma: {self.gamma}
  GAE Lambda: {self.gae_lambda}
  Clip Range: {self.clip_range}
  Entropy Coefficient: {self.ent_coef}

Training Settings:
  Device: {self.device}
  Parallel Envs: {self.n_envs}
  Total Timesteps: {self.total_timesteps:,}
  Save Frequency: {self.save_freq:,}
  Eval Frequency: {self.eval_freq:,}

Paths:
  Model Save Path: {self.model_save_path}
  Log Path: {self.log_path}
  TensorBoard Log: {self.tensorboard_log}
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