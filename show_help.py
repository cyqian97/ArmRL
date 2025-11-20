"""
Quick script to show command line help without loading dependencies
"""
import argparse

parser = argparse.ArgumentParser(
    description='Train PPO agent with image observations',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Use default config
  python test_env.py

  # Use preset config
  python test_env.py --preset fast
  python test_env.py --preset high_quality

  # Custom training with command line args
  python test_env.py --n_envs 16 --camera_height 64 --total_timesteps 1000000

  # CPU-only training
  python test_env.py --device cpu --n_envs 8

  # Maximum speed training (with 64 CPU cores)
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

parser.print_help()
