import yaml

from scripts.envs.env_params import EnvParams

from scripts.spawnables.robot import Robot
from scripts.spawnables.urdf import Urdf
from scripts.spawnables.obstacle import *

from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance
from scripts.rewards.timesteps import ElapsedTimesteps
from scripts.rewards.collision import Collision
from scripts.rewards.shaking import Shaking

from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.resets.collision_reset import CollisionReset
from scripts.resets.boundary_reset import BoundaryReset

from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
import torch as th

ALGO_MAP = {
    "PPO": (PPO, "MultiInputPolicy"),
    "TD3": (TD3, "MultiInputPolicy"),
    "SAC": (SAC, "MultiInputPolicy"),
    "A2C": (A2C, "MultiInputPolicy"),
    "DDPG": (DDPG, "MultiInputPolicy")
}


def parse_config(path: str, engine:str, isEval:bool):
    # load yaml config file from path
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

        # Extract environment parameters
        env = {}
        params = config.get('env', {})
        env["engine"] = "PyBullet"  if engine == "pybullet" else "Isaac"
        env["robots"] = [] if "robots" not in params else [_parse_robot(obj) for obj in params["robots"]]
        env["obstacles"] = [] if "obstacles" not in params else [_parse_obstacle(obj) for obj in params["obstacles"]]
        env["urdfs"] = [] if "urdfs" not in params else [_parse_urdf(obj) for obj in params['urdfs']]
        env["rewards"] = [] if "rewards" not in params else [_parse_reward(obj) for obj in params["rewards"]]
        env["resets"] = [] if "resets" not in params else [_parse_reset(obj) for obj in params["resets"]]
        env["step_size"] = 0.01 if "step_size" not in params else params["step_size"]
        env["step_count"] = 1 if "step_count" not in params else params["step_count"]
        env["headless"] = True if "headless" not in params else params["headless"] 
        env["num_envs"] = 1 if "num_envs" not in params else params["num_envs"] 
        env["env_offset"] = [4, 4] if "env_offset" not in params else params["env_offset"]

        # Extract runtime parameters
        run = {}
        params = config.get('run', {})

        # general run parameters of the model
        run["path"] = path
        run["engine"] = env["engine"]
        run["load_model"] = False if "load_model" not in params else params["load_model"]
        run["model_name"] = None if "model_name" not in params else params["model_name"]
        run["checkpoint"] = None if "checkpoint" not in params else params["checkpoint"]
        
        # Extract specific model paramerters
        custom_algo = params.get('algorithm', {})

        # use default algorithm if not defined by config file
        if len(custom_algo) == 0:
            run["algorithm"] = TD3
            run["policy"] = "MultiInputPolicy"
            run["learning_rate"] = 0.0001
            run["parameters"] = {}

        # extract specific parameters for the model
        else:
            run["algorithm"] = ALGO_MAP[custom_algo["type"]][0]
            run["policy"] = ALGO_MAP[custom_algo["type"]][1]

            run["parameters"] = custom_algo.get('parameters', {})

            if "custom_policy" not in custom_algo:
                run["custom_policy"] = None
            else:
                if custom_algo["custom_policy"]["activation_function"] == "ReLU":
                    activation_function = th.nn.ReLU
                elif custom_algo["custom_policy"]["activation_function"] == "Tanh":
                    activation_function = th.nn.Tanh
                else:
                    raise Exception("Unsupported activation function!")
                
                pol_dict = dict(activation_fn=activation_function)
                
                if custom_algo["type"] in ["PPO", "A2C", "TRPO", "AttentionPPO", "RecurrentPPO"]:
                    vf_pi_dict = dict(vf=[], pi=[])
                    q_name = "vf"
                else:
                    vf_pi_dict = dict(qf=[], pi=[])
                    q_name = "qf"
                
                for layer in custom_algo["custom_policy"]["value_function"]:
                    vf_pi_dict[q_name].append(layer)
                
                for layer in custom_algo["custom_policy"]["policy_function"]:
                    vf_pi_dict["pi"].append(layer)    
                
                pol_dict["net_arch"] = vf_pi_dict
                run["custom_policy"] = pol_dict     

        train = {}
        eval = {}
        if isEval:
            # Extract evaluation parameters
            params = config.get('evaluation', {})
            eval["timesteps"] = 15000000 if "timesteps" not in params else params["timesteps"]
            eval["logging"] = 0 if "logging" not in params else params["logging"]
            env["verbose"] = eval["logging"]
            run["verbose"] = eval["logging"]
        else:
            # Extract training parameters
            params = config.get('train', {})
            train["timesteps"] = 15000000 if "timesteps" not in params else params["timesteps"]
            train["logging"] = 0 if "logging" not in params else params["logging"]
            env["verbose"] = train["logging"]
            run["verbose"] = train["logging"]
            
            train["save_freq"] = 30000 if "save_freq" not in params else params["save_freq"]

    return EnvParams(**env), run, train, eval


def _parse_robot(params: dict) -> Robot:
    return Robot(**params)

def _parse_urdf(params: dict) -> Urdf:
    return Urdf(**params)

def _parse_obstacle(params: dict) -> Obstacle:
    selector = {
        "Cube" : Cube,
        "Sphere" : Sphere,
        "Cylinder" : Cylinder
    }

    # extract required type
    obj_type = params["type"]

    # make sure parsing of obstacle type is implemented
    if obj_type not in selector:
        raise Exception(f"Obstacle parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # position, orientation and scale may be specified as range. Since YAML only supports lists, they need to be transformed into a tuple
    _parse_list_as_tuple(params, "position")
    _parse_list_as_tuple(params, "orientation")
    _parse_list_as_tuple(params, "scale")

    # return instance of parsed obstacle
    return selector[obj_type](**params)

def _parse_list_as_tuple(params: dict, key: str):
    value = params.get(key, None)

    # parameter wasn't specified, no need to transform
    if value is None:
        return
    
    # if list of lists was specified in config file, transform it to a tuple
    if type(value[0]) is list:    
        params[key] = tuple(value)

def _parse_reward(params: dict) -> Reward:
    selector = {
        "Distance": Distance,
        "Collision": Collision,
        "ElapsedTimesteps": ElapsedTimesteps,
        "Shaking": Shaking,
    }

    # extract required type
    type = params["type"]

    # make sure parsing of obstacle type is implemented
    if type not in selector:
        raise Exception(f"Reward parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed reward
    return selector[type](**params)

def _parse_reset(params: dict) -> Reset:
    selector = {
        "DistanceReset": DistanceReset,
        "TimestepsReset": TimestepsReset,
        "CollisionReset": CollisionReset,
        "BoundaryReset": BoundaryReset,
    }

    # extract required type
    type = params["type"]

    # make sure parsing of reset type is implemented
    if type not in selector:
        raise Exception(f"Reset parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed reset
    return selector[type](**params)
    