from scripts.resets.reset import Reset
from scripts.rewards.distance import Distance
from typing import Union

class DistanceReset(Reset):
    def __init__(self, distance: Union[Distance, str], min: float, max: float) -> None:
        # save the name of the distance
        if isinstance(distance, Distance):
            self.distance = distance.name
        else:
            self.distance = distance

        # save min and max values of distance
        self.min = min
        self.max = max