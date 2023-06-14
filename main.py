from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from stable_baselines3 import TD3


# start environment from config file
def _config_specified(path: str):
    # create environment
    env = create_env(parse_config(path))

    # setup model
    model = TD3("MlpPolicy", env, train_freq=1)

    # start learning
    model.learn(1000)
    print("Simple example is complete!")


if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('-f', '--file')

    # parse arguments
    args = parser.parse_args()

    # path to config file was specified
    if "file" in args:        
        _config_specified(args.file)
        exit(0)
    
    print("Use -f to specify a file path to the yaml config file")
