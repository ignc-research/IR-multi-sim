from scripts.envs.params.control_type import ControlType
from scripts.envs.params.env_params import EnvParams
from scripts.envs import create_env
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.rewards.timesteps import ElapsedTimesteps
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3 import TD3, HerReplayBuffer

## pick&place params
# success: distance of cube to target
goal_tolerance = 0.1
# min and max bounds of ur5 range
bounds = (np.array([-0.5, -0.5, 0.1]), np.array([0.5, 0.5, 0.1]))
# cube size adjusted to gripper
scale = np.array([0.1, 0.1, 0.1])

# create parameters for environment
params = EnvParams(
    # Type of environment
    "Isaac",
    # add small ground offset to robot to allow all of its joints to rotate freely
    [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 0.1]), observable_joints=["ee_link"], name="R1")],
    # define obstacles
    [
        # Cube which is supposed to be moved
        Cube(
            position=bounds,
            scale=scale,
            name="ToMove",
            color=array([1, 0, 0]),
            static=False
        ),
        # Cube which defines target of movement
        Cube(
            position=bounds,
            scale=scale,
            name="GoalCube",
            color=array([0, 1, 0]),
            collision=False,
        )
    ],
    # define rewards
    [
        # reward the end effector being close to the object which is supposed to be moved
        Distance("ToMove", "R1/ee_link", name="EffectorDistance"),
        # reward the cube which is supposed to be moved being close to the target
        Distance("ToMove", "GoalCube", name="TargetDistance")
    ],
    # reset if max timesteps was exceeded, or once targetCube reached goal
    [TimestepsReset(500), DistanceReset("TargetDistance", goal_tolerance, 1000)], 
    # overwrite default headless parameter
    headless=False,
    # overwrite default control type
    control_type=ControlType.Velocity    
)

# create issac environment
env = create_env(params)

# setup model
model = TD3(
    "MlpPolicy",
    env,
    train_freq=1
)

# start learning
model.learn(10000)