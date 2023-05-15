from abc import ABC, abstractmethod

_rewards = 0

class Reward(ABC):
    """
    Examples: Distance, -elapsed steps
    """

    def __init__(self, minimize: bool, name: str=None) -> None:
        """
        minimize: true if metric is supposed to be minimized, otherwise false
        """
        self.minimize = minimize

        # set default name
        if name is None:
            global _rewards
            name = f"reward{_rewards}"
            _rewards += 1
        self.name = name

    

