from scripts.rewards.reward import Reward


class ElapsedTimesteps(Reward):
    def __init__(self, minimize: bool, name: str = None) -> None:
        """Rewards or punished the model for an amount equal to the elapsed timesteps

        Args:
            minimize (bool): True if number of timesteps is supposed to be minimized, otherwise false
            name (str, optional): Name of the reward. Defaults to None.
        """
        super().__init__(minimize, name)