from typing import List
from torch import Tensor
from spawnable import Spawnable


class Robot(Spawnable):
    def __init__(self, position: Tensor, orientation: Tensor, urdf_path:str, mass: float, color: List[float]) -> None:
        super().__init__(position, mass, color)
        self.orientation = orientation
        self.urdf_path = urdf_path