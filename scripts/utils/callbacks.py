from stable_baselines3.common.callbacks import BaseCallback
from typing import List

# Allows the custom env (ModularEnv) to log current rewards
class InfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # log rewards
        self.logger.record('average_rewards', self.training_env.get_attr('average_rewards')[0])
        self.logger.record('average_success', self.training_env.get_attr('average_success')[0])
        return True
    
class DistanceCallback(InfoCallback):
    def __init__(self, verbose: int = 0, distance_names: List[str] = []):
        self.distance_names = distance_names
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # for each distance
        for name in self.distance_names:
            # get the value of each distance
            key = "distance_" + name
            # and record the current average value
            self.logger.record(key, self.training_env.get_attr(key)[0])
            self.logger.record('average_steps', self.training_env.get_attr('average_steps')[0])
            self.logger.record('average_collision', self.training_env.get_attr('average_collision')[0])
        return super()._on_step()
    
def parse_callback(verbose: int, distance_names: List[str]) -> InfoCallback:
    # log rewards and distances
    if verbose >= 2:
        return DistanceCallback(verbose, distance_names)
    # log rewards
    if verbose == 1:
        return InfoCallback(verbose)
    # log no additional data
    return None