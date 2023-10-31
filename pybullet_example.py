from scripts.envs.params.env_params import EnvParams
from scripts.envs import create_env
from scripts.spawnables.obstacle import *
from scripts.spawnables.robot import Robot
from scripts.rewards.distance import Distance
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.envs.params.control_type import ControlType
import numpy as np
from stable_baselines3 import TD3

# create parameters for environment
params = EnvParams(
    # engine
    "PyBullet", 
    
    # define robots
    [Robot(urdf_path = "robots/ur5/urdf/ur5_with_gripper.urdf", 
           name = "R1",
           position = np.array([0, 0, 0.3]), 
           orientation = np.array([0.0, 0.0, 0.0, 1.0]), 
           observable_joints = ["ee_link"])],
    
    # define obstacles
    [Cube(name="TargetCube",
          position = np.array([0.4, 0.4, 1]), 
          orientation = np.array([1, 0, 0, 0]), 
          color=array([0, 1, 0]), 
          scale=[0.1, 0.1, 0.1], 
          collision=False)],
    
    # define rewards
    [Distance("TargetCube", "R1/ee_link", name="TargetDistance")],
    
    # define reset conditions
    [DistanceReset("TargetDistance", 0, 1.5), TimestepsReset(100)],
    
    # define runtime settings
    headless=False,
    step_count=140,
    #step_count=1,
    step_size= 1./240.,

    # define controlltype of the robot joints
    control_type=ControlType.Position
)

# create issac environment
env = create_env(params)

# setup model
model = TD3("MultiInputPolicy", train_freq=1, env=env)

# start learning
model.learn(1000)
print("Simple example is complete!")