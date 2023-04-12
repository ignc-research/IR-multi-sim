from abc import ABC
from torch import Tensor, zeros
from typing import List

class Spawnable(ABC):
    def __init__(self, position: Tensor, mass: float, color: List[float], collision: bool, observable:bool) -> None:
        """
        position: Beginning position of object
        mass: Mass of object
        color: Color of object
        collision: True if object is able to collide with others, otherwise false
        observable: True if the object position and orientation is included in the observation for RL
        """
        self.position = position
        self.mass = mass
        self.color = color
        self.collision = collision
        self.orientation = zeros(4)  # default orientation of any spawnable object
        self.observable = observable