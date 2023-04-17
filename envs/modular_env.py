from typing import List
from abc import abstractmethod
import gym
from gym.utils import seeding

from rewards.reward import Reward
from spawnables.obstacle import Obstacle
from spawnables.robot import Robot

class ModularEnv(gym.Env):
    def __init__(self, robots: List[Robot], obstacles: List[Obstacle], rewards: List[Reward], action_space: gym.Space, observation_space: gym.Space) -> None:
        self.action_space: action_space
        self.observation_space: observation_space

    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    