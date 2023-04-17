from typing import List
from envs.modular_env import ModularEnv
from rewards.reward import Reward
from spawnables.obstacle import Obstacle
from spawnables.robot import Robot

class Isaac_Env(ModularEnv):
    def __init__(self, robots: List[Robot], obstacles: List[Obstacle], rewards: List[Reward]) -> None:
        super().__init__(robots, obstacles, rewards)

    def step(self, action):
        # todo: does this prevent wrappers from being applied to base env?
        raise "Stepping is handeletd by IsaacVecEnv"
    
    def reset(self):
        raise "Resetting is handeletd by IsaacVecEnv"
    