import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

import logging

from robosuite.utils.log_utils import DefaultLogger

# Create a logger, set console and file levels
logger_obj = DefaultLogger(
    logger_name="robosuite_logs",
    console_logging_level="ERROR",  # set console to DEBUG
    file_logging_level="ERROR",  # optionally, file logging
)

logger = logger_obj.get_logger()
logger.setLevel(logging.ERROR)  # Make sure the logger itself is at DEBUG


class RobosuiteImageWrapper(gym.Env):
    """
    Wrapper to make Robosuite environment compatible with stable-baselines3
    Extracts only the front camera image as observation
    """

    def __init__(
        self,
        env_name="PickPlaceCan",
        robots=["Panda"],
        camera_height=84,
        camera_width=84,
        control_freq=20,
        horizon=200,
        use_object_obs=False,
        use_camera_obs=True,
    ):
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
            control_freq=control_freq,
            horizon=horizon,
            use_object_obs=use_object_obs,
            use_camera_obs=use_camera_obs,
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
            low=0, high=255, shape=(camera_height, camera_width, 3), dtype=np.uint8
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

    def render(self, mode="rgb_array"):
        """Render the environment"""
        if mode == "rgb_array":
            obs = self.env._get_observations()
            return obs[self.camera_name]
        return None

    def close(self):
        """Close the environment"""
        self.env.close()
