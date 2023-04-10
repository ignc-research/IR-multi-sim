from abc import ABC, abstractmethod
from typing import List, Tuple
from spawnables.robot import Robot
from spawnables.obstacle import Obstacle
from engines.engine import Engine

class Task(ABC):
    """
    Tasks contain blueprints of robots, obstacles, sensors and goals.
    """

    def __init__(self, robots: List[Robot], obstacles: List[Obstacle], engine: Engine, num_envs:int, boundaries: Tuple[Tuple[float, float, float], Tuple[float, float, float]]) -> None:
        """
        robots: List of robots spawned in each environment.
        obstacles: Lost of obstacles spawned in each environment.
        num_envs: Number of environments simulated concurrently.
        boundaries: Min and max coordiantes reached by robots and obstacles in each environment. Used to space environments.
        """
        self.robots = robots
        self.obstacles = obstacles
        self.engine = engine
        self.num_envs = num_envs
        self.boundaries = boundaries
        self.sensors = []  # todo: implement sensors

        # set up the simulated environments
        engine.set_up(self)

    @abstractmethod
    def get_observations(self) -> List[List]:
        """
        Retrieves the observations from all environments.
        """
        pass

    @abstractmethod
    def apply_actions(self, actions: List[List]):
        """
        Applies actions to all environments.
        """
        pass

    @abstractmethod
    def get_rewards(self) -> List[float]:
        """
        Calculates rewards for all revironments.
        """
        pass

    @abstractmethod
    def is_done(self) -> List[bool]:
        """
        Checks for all enviroments if they are done.
        """
        pass

    @abstractmethod
    def reset(self, env_indices=List[int]):
        """
        Resets environments with given ids.
        """
        pass