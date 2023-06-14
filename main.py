from scripts.envs.params.config_parser import parse_config
from scripts.spawnables.robot import Robot

print(parse_config('./data/configs/example_full.yaml').__dict__)