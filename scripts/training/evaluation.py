from scripts.envs.params.config_parser import parse_config
from scripts.envs import create_env
from pathlib import Path
from scripts.training.model import setup_model
from scripts.training.callbacks import parse_callback

from stable_baselines3.common.evaluation import evaluate_policy


# start environment from config file
def eval(path: str, timesteps: int):
    # make sure logs and model folder exist
    Path("./data/logs").mkdir(exist_ok=True)
    Path("./data/models").mkdir(exist_ok=True)
    
    # parse user config
    params = parse_config(path)

    # create environment
    env = create_env(params)

    # load model
    model, _ = setup_model(path, False, env)

    # evaluate for desired amount of timesteps
    obs = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print("Step ", i, "; Rewards: ", rewards, "; Dones:", dones)
    
    print("Average Rewards: ", info)
    
    # Print out results and clean up environment
    env.close()
    