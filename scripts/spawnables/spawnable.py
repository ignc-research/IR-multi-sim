from abc import ABC
import numpy as np
from typing import List

_spawnable_objects = 0

class Spawnable(ABC):
    def __init__(self, position: np.ndarray, color: List[float], collision: bool, observable:bool, name:str) -> None:
        """
        position: Beginning position of object
        color: Color of object
        collision: True if object is able to collide with others, otherwise false
        observable: True if the object position and orientation is included in the observation for RL. Must be true if the object is part of the reward function
        name: Name of the object. Allows referencing it in reward and reset condition classes
        """
        # set default name
        if name is None:
            global _spawnable_objects
            name = f"obj{_spawnable_objects}"
            _spawnable_objects += 1

        self.position = position
        self.color = color
        self.collision = collision
        self.orientation = np.array([1, 0, 0, 0])  # default orientation of any spawnable object
        self.observable = observable
        self.name = name