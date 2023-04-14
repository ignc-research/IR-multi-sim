from engines.isaac_engine import IsaacEngine
from spawnables.obstacle import *
from rewards.distance import Distance
import numpy as np
from modular_env import ModularEnv
from stable_baselines3 import TD3

# setup engine
engine = IsaacEngine("./", 1, True)

# setup environment
robots = []
obstacles = [
    Cube(np.array([0, 0, 0]), np.array([0, 0, 0]), [1, 1, 1], 0, [1, 1, 1], "TargetCube"),
    Cube(np.array([0, 0, 0]), np.array([0, 0, 0]), [1, 1, 1], 0, [1, 1, 1], "Cube"),
    Cube(np.array([0, 0, 0]), np.array([0, 0, 0]), [1, 1, 1], 0, [1, 1, 1])
    ]
rewards = [Distance("TargetCube", "Cube"), Distance(obstacles[2], obstacles[0])]

env = ModularEnv(robots, obstacles, rewards, engine, 1, (0, 0))

# setup model
model = TD3("MlpPolicy", env)

# start learning
model.learn(1000)
print("Simple example is complete!")