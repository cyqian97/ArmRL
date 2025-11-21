import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import argparse
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
import imageio
from training_config import TrainingConfig, FastTrainingConfig, HighQualityConfig, CPUOnlyConfig

from env_wrapper import RobosuiteImageWrapper

def make_env(config):
    """Factory function to create environment"""
    def _init():
        return RobosuiteImageWrapper(
            env_name=config.env_name,
            robots=config.robots,
            camera_height=config.camera_height,
            camera_width=config.camera_width,
            horizon=config.horizon
            
        )
    return _init


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train PPO agent with image observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config (PPO)
  python test_env.py

  # Use different RL algorithms
  python test_env.py --algo sac
  python test_env.py --algo td3

  # Use preset config
  python test_env.py --preset fast
  python test_env.py --preset high_quality

  # Combine algorithm with preset
  python test_env.py --algo sac --preset fast

  # Custom training with command line args
  python test_env.py --n_envs 16 --camera_height 64 --total_timesteps 1000000

  # SAC with custom settings
  python test_env.py --algo sac --n_envs 8 --total_timesteps 1000000

  # TD3 with CPU-only
  python test_env.py --algo td3 --device cpu --n_envs 8

  # Maximum speed training
  python test_env.py --n_envs 32 --camera_height 64 --horizon 100 --n_steps 256
        """
    )

    # Preset configurations
    parser.add_argument(
        '--preset',
        type=str,
        choices=['default', 'fast', 'high_quality', 'cpu'],
        default=None,
        help='Use a preset configuration (overrides other args if specified)'
    )

    # Environment settings
    parser.add_argument('--env_name', type=str, default='PickPlaceCan',
                        help='Robosuite environment name (default: PickPlaceCan)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='Episode length (default: 200)')

    # Parallel environments
    parser.add_argument('--n_envs', type=int, default=8,
                        help='Number of parallel environments (default: 8)')

    # Image settings
    parser.add_argument('--camera_height', type=int, default=84,
                        help='Camera image height (default: 84)')
    parser.add_argument('--camera_width', type=int, default=84,
                        help='Camera image width (default: 84)')

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--n_steps', type=int, default=512,
                        help='Steps per environment per update (default: 512)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda (default: 0.95)')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clip range (default: 0.2)')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Entropy coefficient (default: 0.01)')

    # Algorithm
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'sac', 'td3'],
                        help='RL algorithm to use (default: ppo)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')

    # Training
    parser.add_argument('--total_timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')

    # Callbacks
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Save checkpoint frequency (default: 10000)')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluation frequency (default: 5000)')

    # Paths
    parser.add_argument('--model_save_path', type=str, default='./models/',
                        help='Model save directory (default: ./models/)')
    parser.add_argument('--log_path', type=str, default='./logs/',
                        help='Log directory (default: ./logs/)')
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/',
                        help='TensorBoard log directory (default: ./tensorboard_logs/)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Choose configuration based on preset or custom args
    if args.preset == 'fast':
        print("Using FastTrainingConfig preset")
        config = FastTrainingConfig()
    elif args.preset == 'high_quality':
        print("Using HighQualityConfig preset")
        config = HighQualityConfig()
    elif args.preset == 'cpu':
        print("Using CPUOnlyConfig preset")
        config = CPUOnlyConfig()
    elif args.preset == 'default':
        print("Using default TrainingConfig")
        config = TrainingConfig()
    else:
        # Use custom configuration from command line args
        print("Using custom configuration from command line arguments")
        config = TrainingConfig(
            env_name=args.env_name,
            horizon=args.horizon,
            n_envs=args.n_envs,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            device=args.device,
            total_timesteps=args.total_timesteps,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            model_save_path=args.model_save_path,
            log_path=args.log_path,
            tensorboard_log=args.tensorboard_log,
        )

    # Print configuration
    print(config)

    # Create the environment with parallel processes
    print(f"Creating {config.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(config) for _ in range(config.n_envs)])

    # VecTransposeImage transposes images from (H, W, C) to (C, H, W) for CNN
    env = VecTransposeImage(env)

    # Create evaluation environment (single env is fine)
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecTransposeImage(eval_env)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.model_save_path,
        name_prefix=f'{args.algo}_pickplace'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{config.model_save_path}best_model',
        log_path=config.log_path,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False
    )

    # Create the RL model based on the selected algorithm
    algo_name = args.algo.upper()
    print(f"Creating {algo_name} model with CnnPolicy...")

    if args.algo == 'ppo':
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=config.tensorboard_log,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            device=config.device,
        )

    elif args.algo == 'sac':
        # SAC is off-policy, doesn't use n_steps, n_epochs, gae_lambda, clip_range
        model = SAC(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=config.tensorboard_log,
            learning_rate=config.learning_rate,
            buffer_size=100000,  # Replay buffer size
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=0.005,  # Soft update coefficient
            ent_coef='auto',  # Automatic entropy tuning
            device=config.device,
        )

    elif args.algo == 'td3':
        # TD3 is off-policy, uses action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = TD3(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=config.tensorboard_log,
            learning_rate=config.learning_rate,
            buffer_size=100000,  # Replay buffer size
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=0.005,  # Soft update coefficient
            action_noise=action_noise,
            device=config.device,
        )

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save the final model
    model_name = f"{args.algo}_pickplace_final"
    model.save(model_name)
    print(f"Training complete! Model saved as '{model_name}'")

    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    frames = []

    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        # Get the frame for visualization
        frame = env.get_images()[0]
        frame = np.flipud(frame)  # Flip image vertically
        frames.append(frame)

        if dones[0]:
            break

    # Save test video
    if frames:
        video_name = f'{args.algo}_trained_episode.mp4'
        imageio.mimsave(video_name, frames, fps=20)
        print(f"Test video saved as '{video_name}'")

    env.close()