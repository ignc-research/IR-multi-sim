import numpy as np
from scripts.rewards.reward import Reward
from scripts.spawnables.spawnable import Spawnable

from typing import Union


class Distance(Reward):
    def __init__(self, obj1: Union[Spawnable, str], obj2: Union[Spawnable, str], minimize: bool=True) -> None:
        """
        obj1, ob2: References to the objects whose distance shall be measured. 
        The reference may either be a spwanabe object (name will be extracted automatically), or a str referencing an objects name.
        Examples: 
        - Distance(Cube, Cube)
        - Distance("Robot1/end_effector_link", Cube)
        - Distance("Robot1/end_effector_link", "TargetCube")
        Note: All objects referenced in rewards must be observable.
        """

        super().__init__(minimize)

        # parse name of first object
        if isinstance(obj1, Spawnable):
            self.obj1 = obj1.name
        else:
            self.obj1 = obj1

        # parse name of second object
        if isinstance(obj2, Spawnable):
            self.obj2 = obj2.name
        else:
            self.obj2 = obj2


def calc_distance(p1: np.ndarray, p2: np.ndarray):
    return np.linalg.norm(p1 - p2)