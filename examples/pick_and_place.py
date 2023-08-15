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

#todo: implement other random obstacles, implement pick&place
goal_tolerance = 0.1

# create parameters for environment
params = EnvParams(
    # Type of environment
    "Isaac",
    # define robots
    [Robot("robots/ur5/urdf/ur5_with_gripper.urdf", np.array([0, 0, 0.3]), observable_joints=["ee_link"], name="R1")],
    # define obstacles
    [
        # Cube which is supposed to be moved
        Cube(
            position=(np.array([-1, -1, 0.1]), np.array([1, 1, 0.1])),
            scale=(np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])),
            name="ToMove",
            color=array([1, 0, 0]),
            static=False
        ),
        # Cube which defines target of movement
        Cube(
            position=(np.array([-1, -1, 0]), np.array([1, 1, 1.5])),
            scale=(np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])),
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
    [TimestepsReset(50), DistanceReset("TargetDistance", goal_tolerance, 1000)], 
    # overwrite default headless parameter
    headless=False,
    # overwrite default control type
    control_type=ControlType.VELOCITY,
    # todo: remove debug parameters below
    num_envs=1
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
model.learn(5000)
print("Simple example is complete!")