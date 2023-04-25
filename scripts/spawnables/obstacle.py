from typing import List, Dict
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable

class Obstacle(Spawnable):
    def __init__(self, position: ndarray, mass: float, color: List[float], collision: bool, observable:bool, name: str=None) -> None:
        super().__init__(position, mass, color, collision, observable, name)


class Cube(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]), 
        orientation: ndarray = array([1, 0, 0, 0]),
        scale: List[float] = [1., 1., 1.],
        mass: float = 1.,
        color: List[float] = [1., 1., 1., 1.],
        collision: bool = True,
        observable: bool = True,
        name: str=None
    )-> None:
        super().__init__(position, mass, color, collision, observable, name)
        self.orientation = orientation
        self.scale = scale

class Sphere(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]),
        radius: float = 1.,
        mass: float = 1.,
        color: List[float] = [1., 1., 1., 1.],
        collision: bool = True,
        observable: bool = True,
        name: str=None
    ) -> None:
        super().__init__(position, mass, color, collision, observable, name)
        self.radius = radius

class Cylinder(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]),
        radius: float = 1.,
        height: float = 2.,
        mass: float = 1.,
        color: List[float] = [1., 1., 1., 1.],
        collision: bool = True,
        observable: bool = True,
        name: str=None
    ) -> None:
        super().__init__(position, mass, color, collision, observable, name)
        self.radius = radius
        self.height = height