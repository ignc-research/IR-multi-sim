import numpy as np
from scripts.rewards.reward import Reward
from scripts.spawnables.spawnable import Spawnable

from typing import Union, Tuple


class Distance(Reward):
    def __init__(self, obj: Union[Spawnable, str], target: Union[Spawnable, str], distance_weight: float=-1, orientation_weight: float=-0.1, exponent: float=1, name: str=None) -> None:
        """
        obj1, ob2: References to the objects whose distance shall be measured. 
        distance_weight: Factor the distance (space) is multiplied with to calculate the reward
        orientation_weight: Factor the distance (orientation) is multiplied with to calculate the reward
        exponent: Exponent applied to the final result. reward=distance^exponent
        normalize: Reward will be in range [0, factor], depending on the current position relative to the beginning position
        The reference may either be a spwanabe object (name will be extracted automatically), or a str referencing an objects name.
        Examples: 
        - Distance(Cube, Cube)
        - Distance("Robot1/end_effector_link", Cube)
        - Distance("Robot1/end_effector_link", "TargetCube")
        Note: All objects referenced in rewards must be observable.
        """

        super().__init__(name)

        # parse name of first object
        if isinstance(obj, Spawnable):
            self.obj = obj.name
        else:
            self.obj = obj

        # parse name of second object
        if isinstance(target, Spawnable):
            self.target = target.name
        else:
            self.target = target

        self.distance_weight = distance_weight
        self.orientation_weight = orientation_weight
        self.exponent = exponent

def calc_distance(p1: np.ndarray, p2: np.ndarray, o1: np.ndarray, o2: np.ndarray) -> Tuple[float, float]:
    """Calculates the distance in position and orientation between two points

    Args:
        p1 (np.ndarray): Point in space: x,y,z coordinate
        p2 (np.ndarray): Point in space: x,y,z coordinate
        o1 (np.ndarray): Orientation in space, quaternion (length 4)
        o2 (np.ndarray): Orientation in space, quaternion (length 4)

    Returns:
        float: distance in float
    """
    # calculate distance (space and rotation)
    distance_space = np.linalg.norm(p1 - p2)
    # ISAAC GYM uses quaternions as orientation # todo: outputs floar scaling with angle, but not angle itself
    distance_rotation = np.arccos(2 * (o1 @ o2.T) ** 2 - 1)

    return distance_space, distance_rotation