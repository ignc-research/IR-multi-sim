from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from os.path import exists
import re
from scripts.envs.modular_env import ModularEnv

MODEL_DIR = "./data/models/"
LOG_DIR = "./data/logs/"

def setup_model(config: dict, env: ModularEnv) -> (BaseAlgorithm, str):
    # try loading existing model
    if config["load_model"]:
        model_path = MODEL_DIR + config["load_name"]
        if exists(model_path):
            model = config["algorithm"].load(model_path, env=env)
            model.set_parameters(model_path)
            print("Loaded existing parameters from", model_path)

            print("Model timesteps:", model.num_timesteps)

            return model, MODEL_DIR + config["save_name"]
        
        print(f"No parameters found at {model_path}!")
        exit(0)

    # create new model if necessary
    else:
        # parse model path for windows(\) and linux(/)
        pattern = r'[\\/]([^\\/]+)\.yaml$'
        match = re.search(pattern, config["path"])
        
        config_name = match.group(1)
        model_path = MODEL_DIR + config_name


        # to hanlde necessary parameter trian_freq only for td3 that does not exist in ppo
        if config["train_freq"]:
            model = config["algorithm"](
                config["policy"],
                env=env,   
                policy_kwargs=config["custom_policy"],   
                verbose=1,   
                tensorboard_log= LOG_DIR + config_name,     
                learning_rate=config["learning_rate"],  
                batch_size=config["batch_size"],
                train_freq=config["train_freq"]       
            )
        else:
            model = config["algorithm"](
                config["policy"],
                env=env,   
                policy_kwargs=config["custom_policy"],   
                verbose=1,   
                tensorboard_log= LOG_DIR + config_name,     
                learning_rate=config["learning_rate"],  
                batch_size=config["batch_size"]  
            )

    return model, model_path