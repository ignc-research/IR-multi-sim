from typing import List
from torch import Tensor
from spawnable import Spawnable

class Obstacle(Spawnable):
    def __init__(self, position: Tensor, mass: float, color: List[float], collision: bool) -> None:
        super().__init__(position, mass, color, collision)

class Cube(Obstacle):
    def __init__(self, position: Tensor, orientation: Tensor, scale: List[float],mass: float, color: List[float]) -> None:
        super().__init__(position, mass, color)
        self.orientation = orientation
        self.scale = scale

class Sphere(Obstacle):
    def __init__(self, position: Tensor, radius: float, mass: float, color: List[float], collision: bool) -> None:
        super().__init__(position, mass, color, collision)
        self.radius = radius

class Cylinder(Obstacle):
    def __init__(self, position: Tensor, radius: float, height: float, mass: float, color: List[float], collision: bool) -> None:
        super().__init__(position, mass, color, collision)
        self.radius = radius
        self.height = height