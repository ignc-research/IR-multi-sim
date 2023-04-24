from envs.isaac_env import IsaacEnv
from spawnables.obstacle import *
from rewards.distance import Distance
import numpy as np
from stable_baselines3 import TD3

# setup environment
robots = []
obstacles = [
    Cube(np.array([0, 0, 0]), name="TargetCube"),
    Cube(np.array([0, 1, 0]), name="Cube"),
    Cube(np.array([2, 2, 0]))
    ]
rewards = [Distance("TargetCube", "Cube"), Distance(obstacles[2], obstacles[0])]

env = IsaacEnv("./", 1, False, robots, obstacles, rewards, 1, (10, 10))

# setup model
model = TD3("MlpPolicy", env)

# start learning
model.learn(1000)
print("Simple example is complete!")