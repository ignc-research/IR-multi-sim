from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from os.path import exists
import signal
import re
from scripts.envs.modular_env import ModularEnv

MODEL_DIR = "./data/models/"
LOG_DIR = "./data/logs/"

def setup_model(config: dict, env: ModularEnv) -> (BaseAlgorithm, str):
    # parse model path for windows(\) and linux(/)
    pattern = r'[\\/]([^\\/]+)\.yaml$'
    match = re.search(pattern, config["path"])
    
    config_name = match.group(1)
    model_path = MODEL_DIR + config_name + ".zip"

    # try loading existing model
    if config["load_model"] and exists(model_path):
        model = config["algorithm"].load(model_path, env=env)
        model.set_parameters(model_path)
        print("Loaded existing parameters from", model_path)

    # create new model if necessary
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