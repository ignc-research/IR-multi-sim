from typing import List, Tuple, Union
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable


class Urdf(Spawnable):
    """ A general class for any type of urdf """
    def __init__(self,
        urdf_path:str,
        name: str = None,
        position: Union[ndarray, List] = array([0, 0, 0]), 
        orientation: Union[ndarray, List] = array([1, 0, 0, 0]),
        scale: Union[ndarray, List] = array([1, 1, 1]),
        observable: bool = False,
        collision: bool = False,
        static: bool = False
        )-> None:

        """
        position: Beginning position of robot.
        orientation: Beginning orientation of robot, as a quaternion.
        urdf_path: Relative urdf file path from project root.
        collision: True if the robot is supposed to collide with surroundings.
        name: Name of the urdf. Defaults to None.
        """
        super().__init__(position, orientation, collision, observable, name)
        
        self.urdf_path = urdf_path
        self.static = static
        self.scale = scale
