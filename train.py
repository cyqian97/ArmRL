import os
import argparse
import shutil
import imageio
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from env_wrapper import RobosuiteImageWrapper
from config import EnvConfig, AlgConfig, TrainConfig, load_config_from_yaml


def make_env(env_cfg: EnvConfig):
    """Factory function to create environment"""

    def _init():
        return RobosuiteImageWrapper(
            env_name=env_cfg.env_name,
            robots=env_cfg.robots,
            horizon=env_cfg.horizon,
            control_freq=env_cfg.control_freq,
            camera_height=env_cfg.camera_height,
            camera_width=env_cfg.camera_width,
            use_camera_obs=env_cfg.use_camera_obs,
            use_object_obs=env_cfg.use_object_obs,
        )

    return _init


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train RL agent with image observations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a YAML config file
  python train.py --config configs/fast.yaml

  # Use different presets
  python train.py --config configs/high_quality.yaml
  python train.py --config configs/cpu_only.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    return parser.parse_args()



def train(config_path):
    """Main training function"""
    # Load configuration from YAML
    env_cfg, alg_cfg, train_cfg, _ = load_config_from_yaml(config_path)
    
    # Create the environment with parallel processes
    print(f"Creating {train_cfg.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(env_cfg) for _ in range(train_cfg.n_envs)])

    # VecTransposeImage transposes images from (H, W, C) to (C, H, W) for CNN
    env = VecTransposeImage(env)

    # Create evaluation environment (single env is fine)
    eval_env = DummyVecEnv([make_env(env_cfg)])
    eval_env = VecTransposeImage(eval_env)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.save_freq,
        save_path=train_cfg.model_save_path,
        name_prefix=f"{alg_cfg.alg_name}_{env_cfg.env_name}",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{train_cfg.model_save_path}best_model",
        log_path=train_cfg.log_path,
        eval_freq=train_cfg.eval_freq,
        deterministic=True,
        render=False,
    )

    # Create the RL model based on the selected algorithm
    algo_name = alg_cfg.alg_name.upper()
    print(f"Creating {algo_name} model with {alg_cfg.policy}...")

    if alg_cfg.alg_name.upper() == "PPO":
        model = PPO(
            alg_cfg.policy,
            env,
            verbose=1,
            tensorboard_log=train_cfg.tensorboard_log,
            learning_rate=alg_cfg.learning_rate,
            n_steps=alg_cfg.n_steps,
            batch_size=alg_cfg.batch_size,
            n_epochs=alg_cfg.n_epochs,
            gamma=alg_cfg.gamma,
            gae_lambda=alg_cfg.gae_lambda,
            clip_range=alg_cfg.clip_range,
            ent_coef=alg_cfg.ent_coef,
            device=train_cfg.device,
        )

    elif alg_cfg.alg_name.upper() == "SAC":
        # SAC is off-policy, doesn't use n_steps, n_epochs, gae_lambda, clip_range
        model = SAC(
            alg_cfg.policy,
            env,
            verbose=1,
            tensorboard_log=train_cfg.tensorboard_log,
            learning_rate=alg_cfg.learning_rate,
            buffer_size=100000,  # Replay buffer size
            batch_size=alg_cfg.batch_size,
            gamma=alg_cfg.gamma,
            tau=0.005,  # Soft update coefficient
            ent_coef="auto",  # Automatic entropy tuning
            device=train_cfg.device,
        )

    elif alg_cfg.alg_name.upper() == "TD3":
        # TD3 is off-policy, uses action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )

        model = TD3(
            alg_cfg.policy,
            env,
            verbose=1,
            tensorboard_log=train_cfg.tensorboard_log,
            learning_rate=alg_cfg.learning_rate,
            buffer_size=100000,  # Replay buffer size
            batch_size=alg_cfg.batch_size,
            gamma=alg_cfg.gamma,
            tau=0.005,  # Soft update coefficient
            action_noise=action_noise,
            device=train_cfg.device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {alg_cfg.alg_name}")

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    # Copy config file to model save directory
    config_filename = os.path.basename(config_path)
    dest_config_path = os.path.join(train_cfg.model_save_path, config_filename)
    shutil.copy2(config_path, dest_config_path)
    print("Training completed!")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Start training
    train(args.config)
    
    # Test the trained model
    from test_model import run_test
    run_test(args.config)