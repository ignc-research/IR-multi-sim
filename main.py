from scripts.envs.params.config_parser import parse_config

print(parse_config('./data/configs/example.yaml').__dict__)