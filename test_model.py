"""
Test a trained RL model and save a video

Usage:
    python test_model.py --config configs/fast.yaml
    python test_model.py --config configs/high_quality.yaml
"""
import os
import shutil
import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from env_wrapper import make_env_test
from config import EnvConfig, AlgConfig, TrainConfig, load_config_from_yaml


def detect_algorithm(model_path: str):
    """Detect which algorithm was used based on model path"""
    model_name = model_path.lower()
    if "ppo" in model_name:
        return "PPO", PPO
    elif "sac" in model_name:
        return "SAC", SAC
    elif "td3" in model_name:
        return "TD3", TD3
    else:
        # Default to PPO if can't detect
        print("Warning: Could not detect algorithm from filename, assuming PPO")
        return "PPO", PPO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test a trained RL model and save video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with config file
  python test_model.py --config configs/fast.yaml

  # Test with different config
  python test_model.py --config configs/high_quality.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    return parser.parse_args()



def run_test(config_path):
    """Run testing of a trained model"""
    # Load configuration from YAML
    env_cfg, alg_cfg, train_cfg, test_cfg = load_config_from_yaml(config_path)
    
    # Check if model file exists
    if not os.path.exists(test_cfg.model_path):
        print(f"Error: Model file '{test_cfg.model_path}' not found!")
        return

    # Detect algorithm from model path
    algo_name, AlgoClass = detect_algorithm(test_cfg.model_path)
    print(f"Detected algorithm: {algo_name}")

    # Generate video filename
    model_basename = os.path.splitext(os.path.basename(test_cfg.model_path))[0]
    video_filename = os.path.join(test_cfg.result_save_path, f"{model_basename}.mp4")

    # Create video directory if needed
    os.makedirs(test_cfg.result_save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing Model: {test_cfg.model_path}")
    print(f"Algorithm: {algo_name}")
    print(f"Environment: {env_cfg.env_name}")
    print(f"Episodes: {test_cfg.n_episodes}")
    print(f"Image Size: {env_cfg.camera_height}x{env_cfg.camera_width}")
    print(f"Output Video: {video_filename}")
    print(f"{'='*60}\n")

    # Load the trained model
    print("Loading model...")
    model = AlgoClass.load(test_cfg.model_path, device=test_cfg.device)
    print("Model loaded successfully!")

    # Create environment with offscreen renderer enabled for video recording
    print("\nCreating environment...")
    env = make_env_test(env_cfg, force_offscreen_renderer=test_cfg.save_video)
    print("Environment created!")

    # Test the model
    print(f"\nRunning {test_cfg.n_episodes} test episodes...")
    all_frames = []
    episode_rewards = []
    episode_lengths = []

    for episode in range(test_cfg.n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        frames = []

        print(f"\nEpisode {episode + 1}/{test_cfg.n_episodes}:")

        for step in range(env_cfg.horizon):
            # Predict action
            action, _states = model.predict(obs, deterministic=test_cfg.deterministic)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get frame for video by rendering the environment
            if test_cfg.save_video:
                # Render from the environment (works for both camera and object obs)
                frame = env.env.sim.render(
                    camera_name="frontview",
                    height=test_cfg.video_height,
                    width=test_cfg.video_width
                )
                frame = np.flipud(frame)  # Flip image vertically
                frames.append(frame)

            episode_reward += reward
            episode_length += 1

            # Print progress every 50 steps
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{env_cfg.horizon} | Reward: {episode_reward:.2f}")

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_frames.extend(frames)

        print(
            f"  Episode {episode + 1} complete: Reward = {episode_reward:.2f}, Length = {episode_length}"
        )

    # Close environment
    env.close()

    # Print statistics
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Episodes:           {test_cfg.n_episodes}")
    print(f"Average Reward:     {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min/Max Reward:     {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Average Length:     {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Total Frames:       {len(all_frames)}")
    print(f"{'='*60}\n")

    # Save video
    if test_cfg.save_video and all_frames:
        print(f"Saving video to '{video_filename}'...")
        try:
            imageio.mimsave(video_filename, all_frames, fps=test_cfg.video_fps)
            print("Video saved successfully!")
            print(f"  File: {video_filename}")
            print(f"  Size: {os.path.getsize(video_filename) / 1024 / 1024:.2f} MB")
            print(f"  Duration: {len(all_frames) / test_cfg.video_fps:.1f} seconds")
        except Exception as e:
            print(f"Error saving video: {e}")

    print("\nDone!")

    # Copy config file to the test result directory
    config_filename = os.path.basename(config_path)
    dest_config_path = os.path.join(test_cfg.result_save_path, config_filename)
    shutil.copy2(config_path, dest_config_path)
    print("Testing completed!")

if __name__ == "__main__":
    args = parse_args()
    run_test(args.config)