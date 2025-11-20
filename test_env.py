import os
os.environ["MUJOCO_GL"] = "egl" 
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import imageio

# Load controller config for the Panda arm
# controller_config = load_part_controller_config(default_controller="OSC_POSE")
controller_config = load_composite_controller_config(controller="BASIC")

# Create the environment
env = suite.make(
    "PickPlaceCan",
    robots=["Panda"],                       # single Panda robot arm
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    # env_configuration="opposed",            # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                    # provide image observations to agent
    camera_names="frontview",               # use "frontview" camera for observations
    camera_heights=256,                      # image height
    camera_widths=256,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)

# Reset environment
obs = env.reset()

# The observation will be a dictionary containing:
# - 'frontview_image': RGB image from front camera (256x256x3)
# - Robot state information (joint positions, velocities, etc.)

print("Observation keys:", obs.keys())
print("Front camera image shape:", obs['frontview_image'].shape)

# this example assumes an env has already been created, and performs one agent rollout
import numpy as np

def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env.action_spec
    return np.random.uniform(low, high)


# reset the environment to prepare for a rollout
obs = env.reset()

# List to store frames
frames = []

done = False
ret = 0.
while not done:
    action = get_policy_action(obs)         # use observation to decide on an action
    obs, reward, done, _ = env.step(action) # play action
    frame = obs['frontview_image']
    frame = np.flipud(frame)
    frames.append(frame)
    ret += reward
env.close()

print("rollout completed with return {}".format(ret))

# Save frames as video
imageio.mimsave('episode.mp4', frames, fps=20)
print("Video saved as episode.mp4")