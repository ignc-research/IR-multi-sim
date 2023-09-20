from typing import List, Tuple
from scripts.spawnables.robot import Robot
from scripts.spawnables.obstacle import Obstacle
from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance
from scripts.resets.reset import Reset
from scripts.envs.params.control_type import ControlType

class EnvParams():
    def __init__(
            self,
            engine: str,
            robots: List[Robot],
            obstacles: List[Obstacle],
            rewards: List[Reward],
            resets: List[Reset],
            asset_path: str="./data",
            step_size: float=0.01,
            step_count: int=1,
            headless: bool=True,
            num_envs: int=8,
            env_offset: Tuple[float, float]=(10, 10),
            control_type: ControlType=ControlType.Position,
            max_velocity: int = 5,
            verbose: int = 1
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
        control_type: Type of robot control by ML algorithm
        verbose: verbocity level: 0 for no log, 1 for rewards, 2 for rewards and distances
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
        self.control_type = control_type
        self.max_velocity = max_velocity
        self.verbose = verbose

        # extract information about obstacles
        self.obstacle_count = len(obstacles)
        self.observable_obstacles_count = len([o for o in obstacles if o.observable])

        # extract information about robots
        self.robot_count = len(robots)
        self.observable_robots_count = len([r for r in robots if r.observable])
        self.observable_robot_joint_count = sum(len(r.observable_joints) for r in robots)

        # extract information about distance rewards
        self.distance_count = len(self.get_distance_rewards())
    
    def get_distance_rewards(self) -> List[Distance]:
        distances = []

        for reward in self.rewards:
            if isinstance(reward, Distance):
                distances.append(reward)

        return distances
    
    def get_distance_names(self) -> List[str]:
        return [d.name for d in self.get_distance_rewards()]