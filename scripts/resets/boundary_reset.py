from typing import Union

from scripts.resets.reset import Reset
from scripts.spawnables.spawnable import Spawnable
import numpy as np

class BoundaryReset(Reset):
    def __init__(self, obj: Union[Spawnable, str], min_bound: np.ndarray, max_bound: np.ndarray, reward: float=0) -> None:
        super().__init__(reward)

        # parse name of object
        if isinstance(obj, Spawnable):
            self.obj = obj.name
        else:
            self.obj = obj

        self.min_bound = min_bound
        self.max_bound = max_bound