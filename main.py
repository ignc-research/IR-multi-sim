from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from stable_baselines3 import TD3
from os.path import exists
import signal


# start environment from config file
def _read_config(path: str, timesteps: int, load_model: bool):
    # parse model path
    config_name = path.split('/')[-1].replace('.yaml', '')
    model_path = "./data/models/" + config_name + ".zip"
    
    # create environment
    env = create_env(parse_config(path))

    # try loading existing model
    if load_model and exists(model_path):
        model = TD3.load(model_path, env=env)
        model.set_parameters(model_path)
        print("Loaded existing parameters from", model_path)

    # create new model
    else:
        model = TD3("MlpPolicy", env, train_freq=1, tensorboard_log="./data/logs/"+config_name)
        print(f"No parameters found at {model_path}, creating new model!")

    # default behaviour of program on exit
    def on_exit():
        print(f"Terminated execution after {model.num_timesteps} timesteps")
        model.save(model_path)
        env.close()

    # function executed if program gets interrupted
    def on_interrupt(sig, frame):
        on_exit()
        exit(0)

    # todo: use logger or wrapper:Monitor to log success rate

    # save model if program gets interrupted
    signal.signal(signal.SIGINT, on_interrupt)

    # start learning
    model.learn(timesteps)

    # conclude program execution
    on_exit()


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
