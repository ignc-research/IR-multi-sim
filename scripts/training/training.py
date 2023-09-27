from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from pathlib import Path
from scripts.training.model import setup_model
from scripts.training.callbacks import parse_callback


# start environment from config file
def train(path: str, timesteps: int, load_model: bool):
    # make sure logs and model folder exist
    Path("./data/logs").mkdir(exist_ok=True)
    Path("./data/models").mkdir(exist_ok=True)
    
    # parse user config
    params = parse_config(path)

    # create environment
    env = create_env(params)

    # load or create model
    model, model_path = setup_model(path, load_model, env, params)

    # learn for desired amount of timesteps
    model.learn(timesteps, callback=parse_callback(params.verbose, params.get_distance_names()))

    # learning done: safe parameters and clean up environment
    model.save(model_path)
    env.close()