import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
import imageio
from training_config import TrainingConfig, FastTrainingConfig, HighQualityConfig


class RobosuiteImageWrapper(gym.Env):
    """
    Wrapper to make Robosuite environment compatible with stable-baselines3
    Extracts only the front camera image as observation
    """
    def __init__(self, env_name="PickPlaceCan", robots=["Panda"],
                 camera_height=84, camera_width=84, horizon=200):
        super(RobosuiteImageWrapper, self).__init__()

        # Load controller config
        controller_config = load_composite_controller_config(controller="BASIC")

        # Create the robosuite environment
        self.env = suite.make(
            env_name,
            robots=robots,
            gripper_types="default",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=horizon,
            use_object_obs=False,
            use_camera_obs=True,
            camera_names="frontview",
            camera_heights=camera_height,
            camera_widths=camera_width,
            reward_shaping=True,
        )

        # Define action space
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define observation space (image: H x W x 3, RGB, values 0-255)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(camera_height, camera_width, 3),
            dtype=np.uint8
        )

        self.camera_name = "frontview_image"

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        obs = self.env.reset()
        image_obs = obs[self.camera_name]
        return image_obs, {}

    def step(self, action):
        """Take a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        image_obs = obs[self.camera_name]

        # Gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False

        return image_obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        """Render the environment"""
        if mode == 'rgb_array':
            obs = self.env._get_observations()
            return obs[self.camera_name]
        return None

    def close(self):
        """Close the environment"""
        self.env.close()


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


if __name__ == "__main__":
    # ===== CHOOSE YOUR CONFIGURATION =====
    # Option 1: Default config (balanced)
    config = TrainingConfig()

    # Option 2: Fast training (use this for quick experiments)
    # config = FastTrainingConfig()

    # Option 3: High quality (use this for final training)
    # config = HighQualityConfig()

    # Option 4: Custom config
    # config = TrainingConfig(n_envs=16, camera_height=64, total_timesteps=1000000)

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
        name_prefix='ppo_pickplace'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{config.model_save_path}best_model',
        log_path=config.log_path,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False
    )

    # Create the PPO model with CNN policy
    print("Creating PPO model with CnnPolicy...")
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

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save the final model
    model.save("ppo_pickplace_final")
    print("Training complete! Model saved as 'ppo_pickplace_final'")

    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    frames = []

    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        # Get the frame for visualization
        frame = env.get_images()[0]
        frames.append(frame)

        if dones[0]:
            break

    # Save test video
    if frames:
        imageio.mimsave('trained_episode.mp4', frames, fps=20)
        print("Test video saved as 'trained_episode.mp4'")

    env.close()