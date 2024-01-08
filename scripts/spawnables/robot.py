from typing import List, Union
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable


class Robot(Spawnable):
    def __init__(
        self,
        urdf_path:str,
        position: Union[ndarray, List] = array([0, 0, 0]), 
        orientation: Union[ndarray, List] = array([1, 0, 0, 0]), 
        collision: bool = True,
        observable: bool = True,
        observable_joints: List[str]=[],
        control_type: str = None,
        max_velocity: float = None,
        name: str = None) -> None:
        """
        position: Beginning position of robot.
        orientation: Beginning orientation of robot, as a quaternion.
        urdf_path: Relative urdf file path from project root.
        color: Color of robot.
        collision: True if the robot is supposed to collide with surroundings.
        observable: True if the robots position and orientation is included in the observations for RL training.
        observable_joints: List of joint names whose relative positions and orientation must be included in observations. Defaults to none.
        name: Name of the robot. Defaults to None.
        """
        super().__init__(position, orientation, collision, observable, name)

        self.urdf_path = urdf_path
        self.observable_joints = observable_joints
        self.control_type = control_type if control_type else "Velocity"
        self.max_velocity = max_velocity