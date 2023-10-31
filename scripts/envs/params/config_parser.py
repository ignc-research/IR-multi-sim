from typing import List
from scripts.envs.params.env_params import EnvParams
import yaml
from scripts.spawnables.robot import Robot
from scripts.spawnables.obstacle import *
from scripts.spawnables.urdf import Urdf
from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance
from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.envs.params.control_type import ControlType


def parse_config(path: str) -> EnvParams:
    # load yaml config file from path
    with open(path, 'r') as file:
        # parse yaml file
        content = yaml.load(file, yaml.SafeLoader)

        # parse required parameters
        content["robots"] = [_parse_robot(params) for params in _parse_params(content["robots"])]
        content["urdfs"] = [_parse_urdf(params) for params in _parse_params(content["urdfs"])]
        content["obstacles"] = [_parse_obstacle(params) for params in _parse_params(content["obstacles"])] 
        content["rewards"] = [_parse_reward(params) for params in _parse_params(content["rewards"])]

        # parsing name is not required since reset conditions have no name
        content["resets"] = [_parse_reset(params) for params in content["resets"].values()]

        # control type is specified as string. Transform it to enum
        _parse_control_type(content)

        # parse optional parameters
        return EnvParams(**content)

def _parse_params(config: dict) -> List[dict]:
    """
    Per default, the key of the given dictionary is the name of the object.
    Extracts the name and adds it as as key
    """
    
    # save name in parameters
    for name, params in config.items():
        params["name"] = name
    
    # return parameters
    return config.values()

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
        "Distance": Distance
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
        "TimestepsReset": TimestepsReset
    }

    # extract required type
    type = params["type"]

    # make sure parsing of obstacle type is implemented
    if type not in selector:
        raise Exception(f"Reset parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed reward
    return selector[type](**params)

def _parse_control_type(params: dict):
    control_str = params.get("control_type", None)

    # no control type was specified, use default
    if control_str is None:
        return
    
    try:
        # transform the str to the enum control type
        params["control_type"] = ControlType[control_str]
    except KeyError:
        raise Exception(f"Unknown control type {control_str}! Supported values: {[e.value for e in ControlType]}")