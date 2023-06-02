from typing import List, Tuple
from scripts.spawnables.robot import Robot
from scripts.spawnables.obstacle import Obstacle
from scripts.rewards.reward import Reward
from scripts.resets.reset import Reset

class EnvParams():
    def __init__(
            self,
            robots: List[Robot],
            obstacles: List[Obstacle],
            rewards: List[Reward],
            resets: List[Reset],
            asset_path: str,
            step_size: float,
            headless: bool,
            num_envs: int,
            env_offset: Tuple[float, float]
        ) -> None:
        self.robots = robots
        self.obstacles = obstacles
        self.rewards = rewards
        self.resets = resets
        self.asset_path = asset_path
        self.step_size = step_size
        self.headless = headless
        self.num_envs = num_envs
        self.env_offset = env_offset