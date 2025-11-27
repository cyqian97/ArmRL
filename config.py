"""
Configuration options for faster PPO training with images

Usage:
	from config import Config
	config = Config()
	# Or with custom values:
	config = Config(n_envs=16, camera_height=64)
	# Or load from YAML:
	env_cfg, alg_cfg, train_cfg = load_config_from_yaml("config.yaml")
"""

import yaml
from pathlib import Path


def load_config_from_yaml(
	yaml_path: str,
) -> tuple["EnvConfig", "AlgConfig", "TrainConfig", "TestConfig"]:
	"""
	Load EnvConfig, AlgConfig, TrainConfig, and TestConfig from a YAML file.

	Args:
		yaml_path: Path to the YAML configuration file

	Returns:
		Tuple of (EnvConfig, AlgConfig, TrainConfig, TestConfig)

	Example YAML format:
		env:
		  env_name: "PickPlaceCan"
		  robots: ["Panda"]
		  horizon: 200
		alg:
		  learning_rate: 0.0003
		  n_steps: 512
		train:
		  n_envs: 8
		  device: "cuda"
		test:
		  model_path: "./models/PPO_pickplace_final.zip"
		  n_episodes: 5
		  save_video: true
	"""
	yaml_path = Path(yaml_path)
	if not yaml_path.exists():
		raise FileNotFoundError(f"Config file not found: {yaml_path}")

	print(f"Loading config from: {yaml_path}\n")
	with open(yaml_path, "r") as f:
		config_dict = yaml.safe_load(f)

	env_cfg = EnvConfig(**config_dict.get("env", {}))
	alg_cfg = AlgConfig(**config_dict.get("alg", {}))
	train_cfg = TrainConfig(**config_dict.get("train", {}))
	test_cfg = TestConfig(**config_dict.get("test", {}))

	print(env_cfg)
	print(alg_cfg)
	print(train_cfg)
	print(
		f"\tSteps per Update: {alg_cfg.n_steps} x {train_cfg.n_envs} = {alg_cfg.n_steps * train_cfg.n_envs:,}"
	)
	print(test_cfg)
	return env_cfg, alg_cfg, train_cfg, test_cfg


class EnvConfig:
	"""Environment configuration settings"""

	def __init__(
		self,
		env_name="PickPlaceCan",
		robots=["Panda"],
		horizon=200,
		control_freq=20,
		use_object_obs=False,
		use_camera_obs=True,
		camera_height=84,
		camera_width=84,
		has_renderer=False,
		has_offscreen_renderer=True,
		reward_shaping=True,
	):
		self.env_name = env_name
		self.robots = robots
		self.horizon = horizon
		self.control_freq = control_freq
		self.use_object_obs = use_object_obs
		self.use_camera_obs = use_camera_obs
		self.camera_height = camera_height
		self.camera_width = camera_width
		self.has_renderer=has_renderer
		self.has_offscreen_renderer=has_offscreen_renderer
		self.reward_shaping=reward_shaping

	def __repr__(self):
		"""Print configuration summary"""
		return f"""
Environment Configuration:
============================================================
	Environment Name: {self.env_name}
	Robots: {self.robots}
	Horizon: {self.horizon}
	Control Frequency: {self.control_freq} Hz
	Camera Height: {self.camera_height}
	Camera Width: {self.camera_width}
	Image Size: {self.camera_height}x{self.camera_width}
	Use Object Obs: {self.use_object_obs}
	Use Camera Obs: {self.use_camera_obs}
	Has Renderer: {self.has_renderer}
	Has Offscreen Renderer: {self.has_offscreen_renderer}
	Reward Shaping: {self.reward_shaping}
"""


class AlgConfig:
	"""Algorithm configuration settings"""

	def __init__(
		self,
		alg_name="PPO",
		policy="CnnPolicy",
		learning_rate=3e-4,
		n_steps=512,
		batch_size=64,
		n_epochs=10,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		ent_coef=0.001,
	):
		self.alg_name = alg_name
		self.policy = policy
		self.learning_rate = learning_rate
		self.n_steps = n_steps
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_range = clip_range
		self.ent_coef = ent_coef
		self.use_sde=True
		self.sde_sample_freq=8

	def __repr__(self):
		"""Print configuration summary"""
		return f"""
Algorithm Configuration:
============================================================
	Algorithm Name: {self.alg_name}
	Policy: {self.policy}
	Learning Rate: {self.learning_rate}
	N Steps: {self.n_steps}
	Batch Size: {self.batch_size}
	N Epochs: {self.n_epochs}
	Gamma: {self.gamma}
	GAE Lambda: {self.gae_lambda}
	Clip Range: {self.clip_range}
	Entropy Coefficient: {self.ent_coef}
	Use SDE Noise: {self.use_sde}
	SDE Sample Freq: {self.sde_sample_freq}
"""


class TrainConfig:
	"""Training configuration settings"""

	def __init__(
		self,
		n_envs=8,
		device="cuda",
		total_timesteps=500_000,
		save_freq=10000,
		eval_freq=5000,
		model_save_path="./models/",
		log_path="./logs/",
		tensorboard_log="./tensorboard_logs/",
	):
		self.n_envs = n_envs
		self.device = device
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
============================================================
	Parallel Envs: {self.n_envs}
	Device: {self.device}
	Total Timesteps: {self.total_timesteps:,}
	Save Frequency: {self.save_freq:,}
	Eval Frequency: {self.eval_freq:,}
	Model Save Path: {self.model_save_path}
	Log Path: {self.log_path}
	TensorBoard Log: {self.tensorboard_log}
"""


class TestConfig:
	"""Configuration for testing/evaluating trained models"""

	def __init__(
		self,
		model_path="",
		n_episodes=5,
		deterministic=True,
		render=False,
		save_video=True,
		result_path="./videos/",
		video_fps=20,
		device="cuda",
	):
		self.model_path = model_path
		self.n_episodes = n_episodes
		self.deterministic = deterministic
		self.render = render
		self.save_video = save_video
		self.result_path = result_path
		self.video_fps = video_fps # Normally should match control frequency
		self.device = device

	def __repr__(self):
		"""Print configuration summary"""
		return f"""
Test Configuration:
============================================================
	Model Path: {self.model_path}
	N Episodes: {self.n_episodes}
	Deterministic: {self.deterministic}
	Render: {self.render}
	Save Video: {self.save_video}
	Video Path: {self.result_path}
	Video FPS: {self.video_fps}
	Device: {self.device}
"""


# ====================
# OPTIMIZATION NOTES
# ====================
"""
Memory Usage:
-------------
- Main thread: MEM~8GB, GPU~3GB
- Each env instance: MEM~4GB, GPU~1GB
"""
if __name__ == "__main__":
	import sys

	if len(sys.argv) < 2:
		print("Usage: python config.py <config.yaml>")
		print("Example: python config.py configs/fast.yaml")
		sys.exit(1)

	yaml_path = sys.argv[1]

	env_cfg, alg_cfg, train_cfg, test_cfg = load_config_from_yaml(yaml_path)