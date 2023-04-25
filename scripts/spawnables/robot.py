from typing import List
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable


class Robot(Spawnable):
    def __init__(
        self,
        urdf_path:str,
        position: ndarray = array([0, 0, 0]), 
        orientation: ndarray = array([1, 0, 0, 0]), 
        mass: float = 1.,
        color: List[float] = [1., 1., 1., 1.],
        collision: bool = True,
        observable: bool = True,
        observable_joints: List[str]=[],
        name: str = None) -> None:
        """
        position: Beginning position of robot.
        orientation: Beginning orientation of robot, as a quaternion.
        urdf_path: Relative urdf file path from project root.
        mass: Mass of robot.
        color: Color of robot.
        collision: True if the robot is supposed to collide with surroundings.
        observable: True if the robots position and orientation is included in the observations for RL training.
        observable_joints: List of joint names whose relative positions and orientation must be included in observations. Defaults to none.
        name: Name of the robot. Defaults to None.
        """
        super().__init__(position, mass, color, collision, observable, name)
        self.urdf_path = urdf_path
        self.orientation = orientation
        self.observable_joints = observable_joints