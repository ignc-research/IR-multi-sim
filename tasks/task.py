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

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def apply_actions(self, actions):
        pass

    @abstractmethod
    def get_rewards(self) -> List[float]:
        pass

    @abstractmethod
    def is_done(self) -> List[bool]:
        pass

    @abstractmethod
    def reset(self, env_indices=List[bool]):
        pass