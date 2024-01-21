from scripts.envs.modular_env import ModularEnv
from scripts.envs.env_params import EnvParams


#from scripts.envs.pybullet.pybullet_env import PybulletEnv

def create_env(params: EnvParams) -> ModularEnv:
    name = params.engine

    if name == "Isaac":
        # Isaac requires specialised python runtime. Only raise import exception if Isaac env is attempted to be used
        try:
            from scripts.envs.isaac.isaac_env import IsaacEnv
            envs = {"Isaac": IsaacEnv}
        except ModuleNotFoundError as e:
            class IssacEnv():
                def __init__(self, params: EnvParams) -> None:
                    raise e
       
    elif name == "PyBullet":
        try:
            from scripts.envs.pybullet.pybullet_env import PybulletEnv
            envs = {"PyBullet": PybulletEnv}
            
        except ModuleNotFoundError as e:
            class PybulletEnv():
                def __init__(self, params: EnvParams) -> None:
                    raise e
        
    # return new environment of specified type
    if name in envs.keys():
        return envs[name](params)
    
    # environment isn't known
    raise ValueError(f"Unknown type of environment {name}. Known environments: {list(envs.keys())}")