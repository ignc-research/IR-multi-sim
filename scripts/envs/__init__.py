from scripts.envs.modular_env import ModularEnv
from scripts.envs.params.env_params import EnvParams

# Isaac requires specialised python runtime. Only raise import exception if Isaac env is attempted to be used
try:
    from scripts.envs.isaac.isaac_env import IsaacEnv
except ModuleNotFoundError as e:
    class IssacEnv():
        def __init__(self, params: EnvParams) -> None:
            raise e

from scripts.envs.pybullet.pybullet_env import PybulletEnv

def create_env(params: EnvParams) -> ModularEnv:
    name = params.engine

    # dictionary of existing environments
    envs = {
        "Isaac": IsaacEnv,
        "PyBullet": PybulletEnv
    }

    # return new environment of specified type
    if name in envs.keys():
        return envs[name](params)
    
    # environment isn't known
    raise ValueError(f"Unknown type of environment {name}. Known environments: {list(envs.keys())}")