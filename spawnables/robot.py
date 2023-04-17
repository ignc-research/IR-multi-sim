from typing import List
from numpy import ndarray
from spawnables.spawnable import Spawnable


class Robot(Spawnable):
    def __init__(self, position: ndarray, orientation: ndarray, urdf_path:str, mass: float, color: List[float], collision: bool, observable: bool,
                  observable_joints: str="") -> None:
        """
        position: Beginning position of robot
        orientation: Beginning orientation of robot
        urdf_path: Relative urdf file path from project root
        mass: Mass of robot
        color: Color of robot
        collision: True if the robot is supposed to collide with surroundings
        observable: True if the robots position and orientation is included in the observations for RL training
        observable_joints: Regex matching joint names whose relative positions and orientation must be included in observations. Defaults to none.
        """
        super().__init__(position, mass, color, collision, observable)
        self.orientation = orientation
        self.urdf_path = urdf_path

        self.observable_joints = observable_joints