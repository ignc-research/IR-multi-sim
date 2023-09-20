from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import *
from gym.utils import seeding
import gym.spaces
from scripts.envs.params.control_type import ControlType
from scripts.envs.params.env_params import EnvParams
import gym


class ModularEnv(VecEnv):
    def __init__(self, params: EnvParams) -> None:
        self.headless = params.headless  # True if the simulation will not be rendered, otherwise false 
        self.step_size = params.step_size  # Amount of time passing each time .step() is called
        self.env_data: List[Dict[str, Any]] = [{} for _ in range(params.num_envs)]  # Env data saved in dicts

        # parse observation space
        num_obs = self.reset()["observation"].size // params.num_envs

        # format allows to use HER buffer
        obs_space = spaces.Dict({
            # bundle environments observations
            "observation": spaces.Box(np.ones(num_obs) * -np.inf, np.ones(num_obs) * np.inf),
            # goals are coded in x,y,z (position)
            "desired_goal": spaces.Box(np.ones(params.distance_count * 3) * -np.inf, np.ones(params.distance_count * 3) * np.inf),
            "archieved_goal": spaces.Box(np.ones(params.distance_count * 3) * -np.inf, np.ones(params.distance_count * 3) * np.inf)
            })
        
        # parse action space after obs space: Reset required to access instanciated robot information
        limits = self._get_action_space(params)
        action_space = spaces.Box(np.array([a[0] for a in limits]), np.array([a[1] for a in limits]))

        # init base class with dynamically created action and observation space
        super().__init__(params.num_envs, obs_space, action_space)

    def _get_action_space(self, params: EnvParams) -> List[Tuple[float, float]]:
        # get joint limits of robots
        dof_limits = self.get_robot_dof_limits()

        # joint positions will be set precisely
        if params.control_type == ControlType.Position:
            return dof_limits
        if params.control_type == ControlType.Velocity:
            # joint limits will be controlled via velocity: Return min/max velocita * joint count
            return [(-params.max_velocity, params.max_velocity) for _ in range(len(dof_limits))]
        
        raise Exception(f"Unknown control type: {params.control_type}")

    @abstractmethod
    def get_robot_dof_limits(self) -> List[Tuple[float, float]]:
        """
        Returns:
            np.ndarray: degrees of freedom position limits. 
            shape is (N, num_dof, 2) where index 0 corresponds to the lower limit and index 1 corresponds to the upper limit.
            Only returns dof limits from the first environments, assuming all other environments contain duplicate robot configurations.
        """
        pass

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        env_util.is_wrapped()

        # per default, modular envs don't support wrappper classes
        return [False for _ in self._get_indices(indices)]
    
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return [getattr(i, method_name)(*method_args, *method_kwargs) for i in self._get_indices(indices)]
    
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return [self.env_data[i][attr_name] for i in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for i in self._get_indices(indices):
            self.env_data[i][attr_name] = value

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_absolute_asset_path(self, urdf_path: str):
        """
        Returns the absolute path to the specified urdf file.
        Note: self.asset_path needs to be set in super class before calling this function.
        """
        return Path(self.asset_path).joinpath(urdf_path)