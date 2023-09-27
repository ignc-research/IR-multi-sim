from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from os.path import exists
import signal
from scripts.envs.modular_env import ModularEnv


def setup_model(path: str, reset: bool, env: ModularEnv) -> (BaseAlgorithm, str):
    # parse model path
    config_name = path.split('/')[-1].replace('.yaml', '')
    model_path = "./data/models/" + config_name + ".zip"

    # try loading existing model
    if not reset and exists(model_path):
        model = TD3.load(model_path, env=env)
        model.set_parameters(model_path)
        print("Loaded existing parameters from", model_path)

    # create new model if necessary
    else:
        model = TD3(
            "MultiInputPolicy",
            train_freq=(32, "step"),
            env=env,
            tensorboard_log="./data/logs/"+config_name,
            verbose=1,
            batch_size=2048
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