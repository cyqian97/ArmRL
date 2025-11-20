"""
Test a trained RL model and save a video

Usage:
    python test_model.py --model ppo_pickplace_final.zip
    python test_model.py --model sac_pickplace_final.zip --episodes 5
    python test_model.py --model models/ppo_pickplace_10000_steps.zip --video my_test.mp4
"""
import os
import numpy as np
import argparse
from stable_baselines3 import PPO, SAC, TD3
import imageio

from env_wrapper import RobosuiteImageWrapper


def detect_algorithm(model_path):
    """Detect which algorithm was used based on model path"""
    model_name = model_path.lower()
    if 'ppo' in model_name:
        return 'ppo', PPO
    elif 'sac' in model_name:
        return 'sac', SAC
    elif 'td3' in model_name:
        return 'td3', TD3
    else:
        # Default to PPO if can't detect
        print("Warning: Could not detect algorithm from filename, assuming PPO")
        return 'ppo', PPO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test a trained RL model and save video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default model
  python test_model.py --model ppo_pickplace_final.zip

  # Test with custom settings
  python test_model.py --model sac_pickplace_final.zip --episodes 5 --video sac_test.mp4

  # Test checkpoint model
  python test_model.py --model models/ppo_pickplace_10000_steps.zip

  # Different environment
  python test_model.py --model ppo_pickplace_final.zip --env_name Stack
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.zip file)'
    )

    parser.add_argument(
        '--env_name',
        type=str,
        default='PickPlaceCan',
        help='Robosuite environment name (default: PickPlaceCan)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of episodes to run (default: 3)'
    )

    parser.add_argument(
        '--camera_height',
        type=int,
        default=84,
        help='Camera height (should match training, default: 84)'
    )

    parser.add_argument(
        '--camera_width',
        type=int,
        default=84,
        help='Camera width (should match training, default: 84)'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=200,
        help='Episode length (default: 200)'
    )

    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Output video filename (default: auto-generated based on model name)'
    )

    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic actions (default: True)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Video FPS (default: 20)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return

    # Detect algorithm from model path
    algo_name, AlgoClass = detect_algorithm(args.model)
    print(f"Detected algorithm: {algo_name.upper()}")

    # Generate video filename if not provided
    if args.video is None:
        model_basename = os.path.splitext(os.path.basename(args.model))[0]
        args.video = f"{model_basename}_test.mp4"

    print(f"\n{'='*60}")
    print(f"Testing Model: {args.model}")
    print(f"Algorithm: {algo_name.upper()}")
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Image Size: {args.camera_height}x{args.camera_width}")
    print(f"Output Video: {args.video}")
    print(f"{'='*60}\n")

    # Load the trained model
    print("Loading model...")
    try:
        model = AlgoClass.load(args.model)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying other algorithms...")
        # Try other algorithms if detection failed
        for name, cls in [('PPO', PPO), ('SAC', SAC), ('TD3', TD3)]:
            try:
                print(f"  Trying {name}...")
                model = cls.load(args.model)
                algo_name = name.lower()
                print(f"✓ Model loaded successfully as {name}!")
                break
            except:
                continue
        else:
            print("Error: Could not load model with any algorithm!")
            return

    # Create environment
    print("\nCreating environment...")
    env = RobosuiteImageWrapper(
        env_name=args.env_name,
        robots=["Panda"],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        horizon=args.horizon
    )
    print("✓ Environment created!")

    # Test the model
    print(f"\nRunning {args.episodes} test episodes...")
    all_frames = []
    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        frames = []

        print(f"\nEpisode {episode + 1}/{args.episodes}:")

        for step in range(args.horizon):
            # Predict action
            action, _states = model.predict(obs, deterministic=args.deterministic)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get frame for video
            frame = obs.copy()
            frame = np.flipud(frame)  # Flip image vertically
            frames.append(frame)

            episode_reward += reward
            episode_length += 1

            # Print progress every 50 steps
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{args.horizon} | Reward: {episode_reward:.2f}")

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_frames.extend(frames)

        print(f"  ✓ Episode {episode + 1} complete: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Close environment
    env.close()

    # Print statistics
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Episodes:           {args.episodes}")
    print(f"Average Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min/Max Reward:     {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Average Length:     {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Total Frames:       {len(all_frames)}")
    print(f"{'='*60}\n")

    # Save video
    print(f"Saving video to '{args.video}'...")
    try:
        imageio.mimsave(args.video, all_frames, fps=args.fps)
        print(f"✓ Video saved successfully!")
        print(f"  File: {args.video}")
        print(f"  Size: {os.path.getsize(args.video) / 1024 / 1024:.2f} MB")
        print(f"  Duration: {len(all_frames) / args.fps:.1f} seconds")
    except Exception as e:
        print(f"Error saving video: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
