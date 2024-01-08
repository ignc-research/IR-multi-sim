from abc import ABC
import numpy as np
from typing import List

_object = 0

class Spawnable(ABC):
    def __init__(self, position: np.ndarray, orientation: np.ndarray, collision: bool, observable:bool, name:str) -> None:
        """
        position: Beginning position of object
        color: Color of object
        collision: True if object is able to collide with others, otherwise false
        observable: True if the object position and orientation is included in the observation for RL. Must be true if the object is part of the reward function
        name: Name of the object. Allows referencing it in reward and reset condition classes
        """
        # set default name
        if name is None:
            global _object
            name = f"obj_{_object+1}"
            _object += 1

        # parse position
        if isinstance(position, List):
            self.position = np.array(position)
        else:
            self.position = position

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = np.array(orientation)
        else:
            self.orientation = orientation

        self.collision = collision
        self.observable = observable
        self.name = name