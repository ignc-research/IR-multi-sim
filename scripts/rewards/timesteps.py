from scripts.rewards.reward import Reward


class ElapsedTimesteps(Reward):
    def __init__(self, weight: float=-1, name: str = None) -> None:
        """Rewards or punished the model for an amount equal to the elapsed timesteps

        Args:
            weight (float): Factor multiplied with number of elapsed timesteps to calculate reward
            name (str, optional): Name of the reward. Defaults to None.
        """
        super().__init__(weight, name)