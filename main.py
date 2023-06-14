from scripts.envs.params.config_parser import parse_config
from scripts.spawnables.robot import Robot

print(parse_config('./data/configs/example.yaml').__dict__)