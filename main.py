from scripts.envs.isaac_env import IsaacEnv
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
import numpy as np
from stable_baselines3 import TD3

# setup environment
robots = [Robot("ur5/urdf/ur5_with_gripper.urdf")]
obstacles = [
    Cube(np.array([0, 0, 0]), name="TargetCube"),
    Cube(np.array([0, 1, 0]), name="Cube"),
    Cube(np.array([2, 2, 0]))
    ]
rewards = [Distance("TargetCube", "Cube"), Distance(obstacles[2], obstacles[0])]

env = IsaacEnv("./data/robots", 1, False, robots, obstacles, rewards, 3, (10, 10))

# setup model
model = TD3("MlpPolicy", env)

# start learning
model.learn(1000)
print("Simple example is complete!")