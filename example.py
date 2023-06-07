from scripts.envs.env_params import EnvParams
from scripts.envs import create_env
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3 import TD3

# create parameters for environment
params = EnvParams(
    # define robots
    [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 0.3]), observable_joints=["ee_link"], name="R1")],
    # define obstacles
    [
        Cube(np.array([0.4, 0.4, 1]), name="TargetCube", color=array([0, 1, 0]), scale=[0.1, 0.1, 0.1], collision=False),
        Sphere(np.array([2, 2, 0.5]), name="Sphere"), Cylinder(np.array([2, 4, 0.5]))
    ],
    # define rewards
    [Distance("TargetCube", "R1/ee_link", name="TargetDistance")],
    # define reset conditions
    [DistanceReset("TargetDistance", 0, 1.5), TimestepsReset(100)],
    # overwrite default headless parameter
    headless=False,
    # overwrite default step count parameter
    step_count=10
)

# create issac environment
env = create_env("Isaac", params)

# setup model
model = TD3("MlpPolicy", env, train_freq=1)

# start learning
model.learn(1000)
print("Simple example is complete!")