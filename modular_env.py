from abc import ABC, abstractmethod
from typing import List, Tuple
from spawnables.robot import Robot
from spawnables.obstacle import Obstacle
from engines.engine import Engine
from stable_baselines3.common.vec_env.base_vec_env import *
from gym.spaces import *
import torch
import numpy as np

from omni.isaac.core.tasks import BaseTask
from omni.isaac.gym.vec_env import VecEnvBase

class ModularEnv(VecEnv):
    """
    Environment containing blueprints of robots, obstacles, sensors and goals.
    """

    def __init__(self, robots: List[Robot], obstacles: List[Obstacle], engine: Engine, num_envs:int, offset: Tuple[float, float]) -> None:
        """
        robots: List of robots spawned in each environment.
        obstacles: List of obstacles spawned in each environment.
        num_envs: Number of environments simulated concurrently.
        offset: Required offset between environments to avoid interaction of independent environments
        """
        assert num_envs > 0, "Must create at least one environment!"

        self.robots = robots
        self.obstacles = obstacles
        self.sensors = []  # todo: implement sensors
        self.engine = engine
        self.offset = offset

        # create buffers for observations, rewards, done, info
        self._obs_buffer = None
        self._rew_buffer = None
        self._done_buffer = None
        self._info_buffer = None

        # set up the simulated environments
        self.robot_ids, self.obstacle_ids, self.sensor_ids = engine.set_up(self)

        # setup super class
        # todo: parse action and observation space from robots, obstacles and sensors
            # todo: get min/max values for each joint of each robot
            # todo: get observation space
        
        num_obs = len(engine.get_observations()[0])
        action_limits = engine.get_robot_dof_limits()

        # super().__init__(num_envs, None, None)
        raise "Not implemented!"

    def step(self, actions) -> VecEnvStepReturn:
        # apply actions to actors
        self.pre_physics_step(actions)

        # step world
        self.engine.step()

        self._obs_buffer = self.engine.get_observations()
        self._rew_buffer = None
        self._done_buffer = None
        self._info_buffer = None

        raise "Not implemented!"
        return self._obs_buffer, self._rew_buffer, self._done_buffer, self._info_buffer

    def reset(self) -> VecEnvObs:
        self.engine.reset()

        # get observations from default poses
        self._obs_buffer = self.engine.get_observations()

        # return updated observation buffer
        return self._obs_buffer

    def pre_physics_step(self, actions):
        raise "Not implemented"
        
        



    