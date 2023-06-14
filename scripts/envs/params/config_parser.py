from typing import List
from scripts.envs.params.env_params import EnvParams
import yaml
from scripts.spawnables.robot import parse_robot
from scripts.spawnables.obstacle import parse_obstacle

def parse_config(path: str) -> EnvParams:
    # load yaml config from path
    with open(path, 'r') as file:
        content = yaml.load(file, yaml.SafeLoader)

        # parse required parameters
        engine = content["engine"]
        robots = [parse_robot(params) for params in _parse_params(content["robots"])]
        obstacles = [parse_obstacle(params) for params in _parse_params(content["obstacles"])]
        rewards = []
        resets = []

        # parse optional parameters
        return EnvParams(engine, robots, obstacles, rewards, resets)

def _parse_params(config: dict) -> List[dict]:
    """
    Per default, the key of the given dictionary is the name of the object.
    Extracts the name and adds it as as key
    """
    
    # save name in parameters
    for name, params in config.items():
        params["name"] = name
    
    # return parameters
    return config.values()

