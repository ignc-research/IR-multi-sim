from scripts.envs.isaac_env import IsaacEnv
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3 import TD3

# setup environment
robots = [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 0.3]), observable_joints=["ee_link"], name="R1")]

# todo: obstacles contain range (min/max) for position and orientation values to allow randomization
obstacles = [
    Cube(np.array([0.4, 0.4, 1]), name="TargetCube", color=array([0, 1, 0]), scale=[0.1, 0.1, 0.1]),
    Sphere(np.array([2, 2, 0.5]), name="Sphere"),
    Cylinder(np.array([2, 4, 0.5]))
    ]

# calculate distance between red target cube and end effector of robot
# todo: fix varying distance in environments
rewards = [Distance("TargetCube", "R1/ee_link", name="TargetDistance")]

resets = [DistanceReset("TargetDistance", 0, 10), TimestepsReset(100)]

env = IsaacEnv("./data", 1, False, robots, obstacles, rewards, resets, 2, (10, 10))

# setup model
model = TD3("MlpPolicy", env, train_freq=1)

# start learning
model.learn(10000)
print("Simple example is complete!")