from scripts.envs.params.env_params import EnvParams
from scripts.envs import create_env
from scripts.spawnables.obstacle import *
from scripts.spawnables.random_obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3 import TD3

# create parameters for environment
params = EnvParams(
    # Type of environment
    "Isaac",
    # define robots
    [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 0.3]), observable_joints=["ee_link"], name="R1")],
    # define obstacles
    [
        Cube(), RandomCube((np.array([0, 0, 0]), np.array([10, 10, 0])), scale=(np.array([0.1, 0.1, 0.1]), np.array([2, 2, 2])))
    ],
    # define rewards
    [],
    # define reset conditions
    [TimestepsReset(10)],
    # overwrite default headless parameter
    headless=False,
    # overwrite default step count parameter
    step_count=10
)

# create issac environment
env = create_env(params)

# setup model
model = TD3("MlpPolicy", env, train_freq=1)

# start learning
model.learn(10000)
print("Simple example is complete!")