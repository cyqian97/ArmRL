```
apt-get update -y
apt-get install -y libglib2.0-0
```

```
conda create -n armrl python=3.12 -y
conda activate armrl
pip install mujoco 
pip install robosuite
pip install "imageio[ffmpeg]"  # For saving videos
pip install opencv-python  # For camera observations
pip install gym
pip install stable-baselines3  # If training RL agents
```