from scripts.resets.reset import Reset
from scripts.rewards.distance import Distance
from typing import Union, Tuple

class DistanceReset(Reset):
    def __init__(self, distance: Union[Distance, str], min_distance:float = None, max_distance: float=2, max_angle: float=180, reward: float=0) -> None:
        super().__init__(reward)

        # save the name of the distance
        if isinstance(distance, Distance):
            self.distance_name = distance.name
        else:
            self.distance_name = distance

        # save bounds
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_angle = max_angle
        