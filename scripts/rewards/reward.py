from abc import ABC

_rewards = 0

class Reward(ABC):
    """
    Examples: Distance, -elapsed steps
    """

    def __init__(self, weight: float, name: str=None) -> None:
        """
        weight: factor the reward is multiplied with
        """
        self.weight = weight

        # set default name
        if name is None:
            global _rewards
            name = f"reward{_rewards}"
            _rewards += 1
        self.name = name

