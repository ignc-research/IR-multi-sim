from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from os.path import exists
import re
from scripts.envs.modular_env import ModularEnv
import os

MODEL_DIR = "./data/models/"
LOG_DIR = "./data/logs/"

def setup_model(config: dict, env: ModularEnv) -> (BaseAlgorithm, str):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # try loading existing model
    if config["load_model"]:
        model_path = MODEL_DIR + config["model_name"] + "/" + config["checkpoint"]
        if exists(model_path):
            model = config["algorithm"].load(model_path, env=env)
            model.set_parameters(model_path)
            print("Loaded existing parameters from", model_path)

            print("Model timesteps:", model.num_timesteps)

            return model, MODEL_DIR + config["model_name"]
        
        print(f"Model not found at {model_path}!")
        exit(0)

    # create new model if necessary
    else:
        # parse model path for windows(\) and linux(/)
        pattern = r'[\\/]([^\\/]+)\.yaml$'
        match = re.search(pattern, config["path"])
        
        config_name = match.group(1)
        model_path = MODEL_DIR + config_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = config["algorithm"](
            config["policy"], 
            env, 
            policy_kwargs=config["custom_policy"], 
            verbose=config["verbose"], 
            tensorboard_log=LOG_DIR + config_name, 
            **config["parameters"]
        )

    return model, model_path