from typing import Union

from scripts.rewards.reward import Reward
from scripts.spawnables.spawnable import Spawnable


class Collision(Reward):
    def __init__(self, obj: Union[Spawnable, str], weight: float=-1, name: str = None) -> None:
        """Rewards or punished the model for collisions made

        Args:
            weight (float): Punishment factor for collisions
            name (str, optional): Name of the reward. Defaults to None.
        """
        super().__init__(name)

        # parse name of object
        if isinstance(obj, Spawnable):
            self.obj = obj.name
        else:
            self.obj = obj

        self.weight = weight
