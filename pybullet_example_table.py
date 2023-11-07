from scripts.envs.params.env_params import EnvParams
from scripts.envs import create_env
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.spawnables.urdf import Urdf
from scripts.rewards.distance import Distance
from scripts.rewards.collision import DetectedCollision
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.resets.collision_reset import CollisionReset
from scripts.envs.params.control_type import ControlType
import numpy as np
from stable_baselines3 import TD3

# create parameters for environment
params = EnvParams(
    # engine
    "PyBullet", 
    
    # define robots
    [
        Robot(urdf_path = "robots/ur5/urdf/ur5_with_gripper.urdf", name = "R1", position = np.array([0, 0, 0.64]), 
           orientation = np.array([1, 0, 0, 0]), observable_joints = ["ee_link"])
    ], 
    
    # define obstacles
    [
        Cube(name="TargetCube", position = (np.array([-0.65, -0.4, 0.63]),np.array([-0.65, 0.4, 1])), collision=False), 
        Sphere(name="Sphere_1", position = np.array([0.4, 0.3, 0.8]), collision=True),
        Sphere(name="Sphere_2", position = np.array([0.4, -0.3, 0.9]), collision=True),
        Cylinder(name="Cylinder_1", position = np.array([-0.45, 0.4, 0.7]), collision=True)
    ],
    
    # define urdfs
    [
        Urdf(urdf_path = "table/table.urdf", type = "Table", name = "Table", position = np.array([0, 0, 0]))
    ],    
    
    # define rewards
    [
        Distance("TargetCube", "R1/ee_link", name="TargetDistance"),
        DetectedCollision("R1", weight=-5, name="CollisionDetection")
    ],
    
    # define reset conditions
    [
        DistanceReset("TargetDistance", 0, 1.5), TimestepsReset(100),
        CollisionReset("R1", 1)
    ],
    
    # define runtime settings
    headless=False,
    step_count=1,
    step_size= 0.00416666666,
    num_envs=4,
    env_offset=[4,4],

    # define controlltype of the robot joints
    control_type=ControlType.Velocity
)

# create issac environment
env = create_env(params)

# setup model
model = TD3("MultiInputPolicy", train_freq=1, env=env)

# start learning
model.learn(1000)
print("Simple example is complete!")