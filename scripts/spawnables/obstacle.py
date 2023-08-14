from typing import List
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable

class Obstacle(Spawnable):
    def __init__(self, position: ndarray, color: List[float], collision: bool, observable:bool, static:bool, name: str=None) -> None:
        super().__init__(position, color, collision, observable, name)
        self.static = static


class Cube(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]), 
        orientation: ndarray = array([1, 0, 0, 0]),
        scale: List[float] = [1., 1., 1.],
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        name: str=None
    )-> None:
        """A cube obstacle spawned in each environment with fixed parameters.

        Args:
            position (ndarray, optional): Position of cube. Defaults to array([0, 0, 0]).
            orientation (ndarray, optional): Orientation of cube in quaternion. Defaults to array([1, 0, 0, 0]).
            scale (List[float], optional): Scale of cube. Defaults to [1., 1., 1.].
            color (ndarray, optional): Color of cube in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of cube, allowing it to be referenced. Defaults to None.
        """
        super().__init__(position, color, collision, observable, static, name)

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = array(orientation)
        else:
            self.orientation = orientation
            
        self.scale = scale

class Sphere(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]),
        radius: float = 1.,
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        name: str=None
    ) -> None:
        """A sphere obstacle spawned in each environment with fixed parameters.

        Args:
            position (ndarray, optional): Position of the sphere. Defaults to array([0, 0, 0]).
            radius (float, optional): Radius of the sphere. Defaults to 1..
            color (ndarray, optional): Color of the sphere in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of sphere, allowing it to be referenced. Defaults to None.
        """
        super().__init__(position, color, collision, observable, static, name)
        self.radius = radius

class Cylinder(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]),
        radius: float = 1.,
        height: float = 2.,
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        name: str=None
    ) -> None:
        """A cylinder obstacle spawned in each environment with fixed parameters.

        Args:
            position (ndarray, optional): Position of the cylinder. Defaults to array([0, 0, 0]).
            radius (float, optional): Radius of the cylinder. Defaults to 1..
            height (float, optional): Height of the cylinder. Defaults to 2..
            color (ndarray, optional): Color of the cylinder in RGB.. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of cylinder, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, static, name)
        self.radius = radius
        self.height = height