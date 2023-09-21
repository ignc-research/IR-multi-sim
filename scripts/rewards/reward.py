from abc import ABC

_rewards = 0

class Reward(ABC):
    """
    Examples: Distance, -elapsed steps
    """

    def __init__(self, name: str=None) -> None:
        # set default name
        if name is None:
            global _rewards
            name = f"reward{_rewards}"
            _rewards += 1
        self.name = name

