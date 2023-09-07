from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from os.path import exists
import signal
from scripts.envs.modular_env import ModularEnv


def _setup_model(path: str, load_model: bool, env: ModularEnv) -> (BaseAlgorithm, str):
    # parse model path
    config_name = path.split('/')[-1].replace('.yaml', '')
    model_path = "./data/models/" + config_name + ".zip"

    # try loading existing model
    if load_model and exists(model_path):
        model = TD3.load(model_path, env=env)
        model.set_parameters(model_path)
        print("Loaded existing parameters from", model_path)
    # create new model if necessary
    else:
        model = TD3("MlpPolicy", env, train_freq=1, tensorboard_log="./data/logs/"+config_name, verbose=1)
        print(f"No parameters found at {model_path}, creating new model!")

    # function executed if program gets interrupted
    def on_interrupt(sig, frame):
        print(f"Terminated execution after {model.num_timesteps} timesteps")
        model.save(model_path)
        env.close()
        exit(0)

    # save model if program gets interrupted
    signal.signal(signal.SIGINT, on_interrupt)

    return model, model_path

# start environment from config file
def _read_config(path: str, timesteps: int, load_model: bool):
    # create environment
    env = create_env(parse_config(path))

    # load or create model
    model, model_path = _setup_model(path, load_model, env)

    # learn for desired amount of timesteps
    model.learn(timesteps)

    # learning done: safe parameters and clean up environment
    model.save(model_path)
    env.close()


if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('-f', '--file', help="Environment config file")
    parser.add_argument('-t', '--timesteps', default=100000, help="Amount of timesteps to train the model")
    parser.add_argument('-l', '--load_model', default=True, help="True if old model parameters are supposed to be loaded")

    # parse arguments
    args = parser.parse_args()

    # path to config file was specified
    if args.file is None:        
        print("Use -f to specify a file path to the yaml config file")
        exit(0)
    
    _read_config(args.file, args.timesteps, args.load_model)
