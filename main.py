from scripts.envs.isaac_env import IsaacEnv
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3 import TD3

# setup environment
robots = [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 1]))]

obstacles = [
    Cube(np.array([0, 0, 0.5]), name="TargetCube", color=array([0, 1, 0])),
    Sphere(np.array([2, 2, 0.5]), name="Sphere"),
    Cylinder(np.array([2, 4, 0.5]))
    ]

# todo: allow calculating distance between joints of robots and robots
rewards = [Distance("TargetCube", "Sphere", name="TargetDistance")]

# todo: reset conditions: Elapsed Timesteps, Reward above/below value
resets = [DistanceReset("TargetDistance", 1, 100), TimestepsReset(100)]

env = IsaacEnv("./data", 1, False, robots, obstacles, rewards, 2, (10, 10))

# setup model
model = TD3("MlpPolicy", env, train_freq=1)

# start learning
model.learn(1000)
print("Simple example is complete!")