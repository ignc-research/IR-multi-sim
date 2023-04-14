from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
from rewards.reward import Reward
from spawnables.robot import Robot
from spawnables.obstacle import Obstacle
from stable_baselines3.common.vec_env.base_vec_env import *
from gym.spaces import *

from omni.isaac.core.tasks import BaseTask
from omni.isaac.gym.vec_env import VecEnvBase

if TYPE_CHECKING:
    from engines.engine import Engine

class ModularEnv(VecEnv):
    """
    Environment containing blueprints of robots, obstacles, sensors and goals.
    TODO: Let engine base class implement VecEnv
    """

    def __init__(self, robots: List[Robot], obstacles: List[Obstacle], rewards: List[Reward], engine: "Engine", num_envs:int, offset: Tuple[float, float]) -> None:
        """
        robots: List of robots spawned in each environment.
        obstacles: List of obstacles spawned in each environment.
        num_envs: Number of environments simulated concurrently.
        offset: Required offset between environments to avoid interaction of independent environments
        """
        assert num_envs > 0, "Must create at least one environment!"

        self.robots = robots
        self.obstacles = obstacles
        self.rewards = rewards
        self.sensors = []  # todo: implement sensors
        self.engine = engine
        self.offset = offset

        # create buffers for observations, rewards, done, info
        self._obs_buffer = None
        self._rew_buffer = None
        self._done_buffer = None
        self._info_buffer = None

        # set up the simulated environments
        engine.set_up(self)

        # setup super class
        # todo: parse action and observation space from robots, obstacles and sensors
            # todo: get min/max values for each joint of each robot
            # todo: get observation space
        
        obs = engine.reset()
        action_limits = engine.get_robot_dof_limits()

        # super().__init__(num_envs, None, None)
        raise "Not implemented!"

    def step(self, actions) -> VecEnvStepReturn:
        return self.engine.step(actions)

    def reset(self) -> VecEnvObs:
        return self.engine.reset()    