from typing import Union

from scripts.resets.reset import Reset
from scripts.spawnables.spawnable import Spawnable

_collReset = 0

class CollisionReset(Reset):
    def __init__(self, obj: Union[Spawnable, str], max: int, reward: float=0, name: str=None) -> None:
        super().__init__(reward)

        # parse name of object
        if isinstance(obj, Spawnable):
            self.obj = obj.name
        else:
            self.obj = obj

        self.max = max
        self.name = name if name else "CollReset_" + f"{_collReset+1}"