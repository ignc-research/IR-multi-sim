from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import *


class ModularEnv(VecEnv):
    def __init__(self, asset_path:str, step_size: float, headless:bool, num_envs: int) -> None:
        self.asset_path = asset_path  # Path to assets used in simulation
        self.headless = headless  # True if the simulation will not be rendered, otherwise false 
        self.step_size = step_size  # Amount of time passing each time .step() is called

        obs = self.reset()
        actions = self.get_robot_dof_limits()
        print(obs, actions)

        raise "Engine constructor not implemented"
        super().__init__(num_envs, None, None)

    @abstractmethod
    def set_joint_positions(
        self,
        positions: Optional[np.ndarray],
        robot_indices: Optional[Union[np.ndarray, List]] = None,
        joint_indices: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Sets the joint positions of all robots specified in robot_indices to their respective values specified in positions.
        """
        pass
    
    @abstractmethod
    def set_joint_position_targets(
        self,
        positions: Optional[np.ndarray],
        robot_indices: Optional[Union[np.ndarray, List]] = None,
        joint_indices: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Sets the joint position targets of all robots specified in robot_indices to their respective values specified in positions.
        """
        pass

    @abstractmethod
    def set_joint_velocities(
        self,
        velocities: Optional[np.ndarray],
        robot_indices: Optional[Union[np.ndarray, List]] = None,
        joint_indices: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Sets the joint velocities of all robots specified in robot_indices to their respective values specified in velocities.
        """
        pass
     
    @abstractmethod   
    def set_joint_velocity_targets(
        self,
        velocities: Optional[np.ndarray],
        robot_indices: Optional[Union[np.ndarray, List]] = None,
        joint_indices: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Sets the joint velocities targets of all robots specified in robot_indices to their respective values specified in velocities.
        """
        pass

    @abstractmethod
    def set_local_poses(
        self,
        translations: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        indices: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Sets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        pass

    @abstractmethod
    def get_local_poses(
        self,
        indices: Optional[Union[np.ndarray, List]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        pass

    @abstractmethod
    def get_collisions(self) -> List[Tuple[str, str]]:
        """
        Returns the ids of objects which are colliding. Updated after each step.
        Example: [(Robot1, Robot2), (Robot1, Obstacle3)] -> Object 1 is colliding with object 2 and 3.
        """
        pass

    @abstractmethod
    def get_robot_dof_limits(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: degrees of freedom position limits. 
            shape is (N, num_dof, 2) where index 0 corresponds to the lower limit and index 1 corresponds to the upper limit.
            Only returns dof limits from the first environments, assuming all other environments contain duplicate robot configurations.

        """
        pass

    def _get_absolute_asset_path(self, urdf_path: str):
        return Path(self.asset_path).joinpath(urdf_path)