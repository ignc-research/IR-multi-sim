from typing import List
from numpy import ndarray
from spawnables.spawnable import Spawnable

class Obstacle(Spawnable):
    def __init__(self, position: ndarray, mass: float, color: List[float], collision: bool, name: str=None) -> None:
        super().__init__(position, mass, color, collision, name)

class Cube(Obstacle):
    def __init__(self, position: ndarray, orientation: ndarray, scale: List[float],mass: float, color: List[float], name: str=None) -> None:
        super().__init__(position, mass, color, name)
        self.orientation = orientation
        self.scale = scale

class Sphere(Obstacle):
    def __init__(self, position: ndarray, radius: float, mass: float, color: List[float], collision: bool, name: str=None) -> None:
        super().__init__(position, mass, color, collision, name)
        self.radius = radius

class Cylinder(Obstacle):
    def __init__(self, position: ndarray, radius: float, height: float, mass: float, color: List[float], collision: bool, name: str=None) -> None:
        super().__init__(position, mass, color, collision, name)
        self.radius = radius
        self.height = height