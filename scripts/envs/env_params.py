from typing import List, Tuple
from scripts.spawnables.robot import Robot
from scripts.spawnables.obstacle import Obstacle
from scripts.rewards.reward import Reward
from scripts.resets.reset import Reset
from typing import Union

class EnvParams():
    def __init__(
            self,
            engine: str,
            robots: List[Robot],
            obstacles: List[Obstacle],
            rewards: List[Reward],
            resets: List[Reset],
            asset_path: str="./data",
            step_size: float=1.0,
            step_count: int=1,
            headless: bool=True,
            num_envs: int=8,
            env_offset: Tuple[float, float]=(10, 10)
        ) -> None:
        """
        engine: Type of engine used to simulate environment.
        robots: Robots simulated in each environment.
        obstacles: Obstacles simulated in each environment.
        rewards: Environment rewards given depending on states of robots and obstacles.
        resets: Environment resets given depending on states of robots and obstacles.
        asset_path: Path to local files, e.g. urdf files of robots.
        step_size: Physics step size of environment.
        step_count: Amount of steps simulated before next control instance.
        headless: True if the simulation will run without visualization, otherwise False.
        num_envs: Number of environments simulated concurrently.
        env_offset: Offset between environments if simulated in the same world.
        """

        # make sure engine is known
        self.engine = engine

        self.robots = robots
        self.obstacles = obstacles
        self.rewards = rewards
        self.resets = resets
        self.asset_path = asset_path
        self.step_size = step_size
        self.step_count = step_count
        self.headless = headless
        self.num_envs = num_envs
        self.env_offset = env_offset