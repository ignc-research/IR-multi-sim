from argparse import ArgumentParser
from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import HerReplayBuffer, PPO
from os.path import exists
from pathlib import Path
import signal
from scripts.envs.modular_env import ModularEnv
from typing import List


# Allows the custom env (ModularEnv) to log current rewards
class InfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # log rewards
        self.logger.record('average_rewards', self.training_env.get_attr('average_rewards')[0])
        return True
    
class DistanceCallback(InfoCallback):
    def __init__(self, verbose: int = 0, distance_names: List[str] = []):
        self.distance_names = distance_names
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # for each distance
        for name in self.distance_names:
            # get the value of each distance
            key = "distance_" + name
            # and record the current average value
            self.logger.record(key, self.training_env.get_attr(key)[0])
        return super()._on_step()
    
def _parse_callback(verbose: int, distance_names: List[str]) -> InfoCallback:
    # log rewards and distances
    if verbose >= 2:
        return DistanceCallback(verbose, distance_names)
    # log rewards
    if verbose == 1:
        return InfoCallback(verbose)
    # log no additional data
    return None

def _setup_model(path: str, reset: bool, env: ModularEnv) -> (BaseAlgorithm, str):
    # parse model path
    config_name = path.split('/')[-1].replace('.yaml', '')
    model_path = "./data/models/" + config_name + ".zip"

    # wrap environment
    # env = TimeFeatureWrapper(env)

    # try loading existing model
    if not reset and exists(model_path):
        model = TQC.load(model_path, env=env)
        model.set_parameters(model_path)
        print("Loaded existing parameters from", model_path)

    # create new model if necessary
    else:
        model = TQC(
            batch_size=2048,
            buffer_size=1000000,
            gamma=0.95,
            learning_rate=0.001,
            policy="MultiInputPolicy",
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict( goal_selection_strategy='future', n_sampled_goal=4),
            tau=0.05,
            env=env,
            tensorboard_log="./data/logs/"+config_name,
            verbose=1,
            learning_starts=1000
        )
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
    # make sure logs and model folder exist
    Path("./data/logs").mkdir(exist_ok=True)
    Path("./data/models").mkdir(exist_ok=True)
    
    # parse user config
    params = parse_config(path)

    # create environment
    env = create_env(params)

    # load or create model
    model, model_path = _setup_model(path, load_model, env)

    # learn for desired amount of timesteps
    model.learn(timesteps, callback=_parse_callback(params.verbose, params.get_distance_names()))

    # learning done: safe parameters and clean up environment
    model.save(model_path)
    env.close()


if __name__ == '__main__':
    parser = ArgumentParser("IR-Multi-Sim", description="Train complex training instructions for machine learning environments with a simple interface")

    # allow parsing file
    parser.add_argument('-f', '--file', help="Environment config file")
    parser.add_argument('-t', '--timesteps', default=1000000, type=int, help="Amount of timesteps to train the model")
    parser.add_argument('-r', '--reset', action="store_true", help="Start model training from scratch")

    # parse arguments
    args = parser.parse_args()

    # path to config file was specified
    if args.file is None:        
        print("Use -f to specify a file path to the yaml config file")
        exit(0)
    
    _read_config(args.file, args.timesteps, args.reset)
