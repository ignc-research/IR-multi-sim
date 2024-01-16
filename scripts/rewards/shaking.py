from typing import Union

from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance
from typing import Union


class Shaking(Reward):
    def __init__(self, distance: Union[Distance, str], weight: float=-0.005, length: int=10, name: str = None) -> None:
        """ Rewards or punished the model for "shaking", 
            meaning beeing stuck in increasing and decreasing distance over and over again 

        Args:
            weight (float): Punishment factor for collisions
            name (str, optional): Name of the reward. Defaults to None.
        """
        super().__init__(name)

        # save the name of the distance
        if isinstance(distance, Distance):
            self.distance_name = distance.name
        else:
            self.distance_name = distance

        self.weight = weight
        self.length = length