from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import *
from gym.utils import seeding

from envs.modular_env import ModularEnv


class ModularVecEnv(VecEnv):
    def __init__(self, asset_path:str, envs: List[ModularEnv]) -> None:
        self.asset_path = asset_path  # Path to assets used in simulation
        self.num_envs = len(envs)  # Number of environments
        self.envs = envs  # Environments

        # extract action and observation space to call base constructor
        env = self.envs[0]
        super().__init__(self.num_envs, env.observation_space, env.action_space)

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass
    
    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
