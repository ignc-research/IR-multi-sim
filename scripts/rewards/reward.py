from abc import ABC, abstractmethod

class Reward(ABC):
    """
    Examples: Distance, -elapsed steps
    """

    def __init__(self, minimize: bool) -> None:
        """
        minimize: true if metric is supposed to be minimized, otherwise false
        """
        self.minimize = minimize

    

