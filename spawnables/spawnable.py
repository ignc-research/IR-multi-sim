from abc import ABC
from torch import Tensor
from typing import List

class Spawnable(ABC):
    def __init__(self, position: Tensor, mass: float, color: List[float], collision: bool) -> None:
        self.position = position
        self.mass = mass
        self.color = color
        self.collision = collision