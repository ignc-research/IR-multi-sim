from abc import ABC

class Reset(ABC):
    def __init__(self, reward: float=0) -> None:
        """Examples: Elapsed timesteps, distance above or below threshhold

        Args:
            reward (float, optional): Reward gained when the reset condition is triggered
        """
        self.reward = reward