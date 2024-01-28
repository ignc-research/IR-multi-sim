from stable_baselines3.common.callbacks import BaseCallback
from typing import List

# Allows the custom env (ModularEnv) to log current rewards
class GeneralCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        self.logger.record('general/avg_rewards', self.training_env.get_attr('avg_rewards')[0])
        self.logger.record('general/avg_successes', self.training_env.get_attr('avg_success')[0])
        self.logger.record('general/avg_resets', self.training_env.get_attr('avg_resets')[0])
        return True
    
class PerformanceCallback(GeneralCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('performance/avg_setupTime', self.training_env.get_attr('avg_setupTime')[0])
        self.logger.record('performance/avg_actionTime', self.training_env.get_attr('avg_actionTime')[0])
        self.logger.record('performance/avg_obsTime', self.training_env.get_attr('avg_obsTime')[0])
        return super()._on_step()

class AdvancedCallback(PerformanceCallback):
    def __init__(self, verbose: int = 0, distance_names: List[str] = []):
        self.distance_names = distance_names
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # for each distance
        for name in self.distance_names:
            # get the value of each distance
            key1 = "avg_" + name + "_euclid_dist" 
            key2 = "avg_" + name + "_anglular_dist" 
            # and record the current average value
            self.logger.record('advanved/avg_dist_euclid_' + name, self.training_env.get_attr(key1)[0])
            self.logger.record('advanved/avg_dist_angular_' + name, self.training_env.get_attr(key2)[0])

        self.logger.record('advanved/avg_steps', self.training_env.get_attr('avg_steps')[0])
        self.logger.record('advanved/avg_coll', self.training_env.get_attr('avg_steps')[0])
        return super()._on_step()

    
def parse_callback(verbose: int, distance_names: List[str]) -> BaseCallback:
    # log distances, collisions, steps
    if verbose > 2:
        return AdvancedCallback(verbose, distance_names)
    # log setup time, action time, obs time
    if verbose == 2:
        return PerformanceCallback(verbose)
    # log rewards, success, resets
    if verbose == 1:
        return GeneralCallback(verbose)
    # log no additional data
    return None