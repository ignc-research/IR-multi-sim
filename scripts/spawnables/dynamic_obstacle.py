from typing import List, Tuple
from numpy import ndarray, array
from scripts.spawnables.obstacle import Obstacle

class DynamicCube(Obstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        orientation: Tuple[ndarray, ndarray] = (array([1, 0, 0, 0]), array([0, 0, 1, 0])),
        scale: Tuple[ndarray, ndarray] = (array([0.5, 0.5, 0.5]), array([1.5, 1.5, 1.5])),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        name: str=None
    )-> None:
        """A cube obstacle spawned in each environment. Each reset, the cubes properties change, as defined by the range specified here.

        Args:
            position (Tuple[ndarray, ndarray], optional): Valid range of positions. Defaults to (array([0, 0, 0]), array([1, 1, 1])).
            orientation (Tuple[ndarray, ndarray], optional): Valid range of orientations. Defaults to (array([1, 0, 0, 0]), array([0, 0, 1, 0])).
            scale (Tuple[ndarray, ndarray], optional): Valid range of scale. Defaults to (array([0.5, 0.5, 0.5]), array([1.5, 1.5, 1.5])).
            color (ndarray, optional): Color of cube in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            name (str, optional): Name of cube, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, name)

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = array(orientation)
        else:
            self.orientation = orientation
            
        self.scale = scale

class DynamicSphere(Obstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        radius: Tuple[float, float] = (0.1, 2.0),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        name: str=None
    ) -> None:
        """A sphere obstacle spawned in each environment. Each reset, the spheres properties change, as defined by the range specified here.

        Args:
            position (Tuple[ndarray, ndarray], optional): Valid range of positions. Defaults to (array([0, 0, 0]), array([1, 1, 1])).
            radius (Tuple[float, float], optional): Valid range of radii. Defaults to (0.1, 2.0).
            color (ndarray, optional): Color of sphere in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flar for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            name (str, optional): Name of sphere, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, name)
        self.radius = radius

class DynamicCylinder(Obstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        radius: Tuple[float, float] = (0.1, 2.0),
        height: Tuple[float, float] = (0.1, 4.0),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        name: str=None
    ) -> None:
        """A cylinder obstacle spawned in each environment. Each reset, the cylinders properties change, as defined by the range specified here.

        Args:
            position (Tuple[ndarray, ndarray], optional): Valid range of positions. Defaults to (array([0, 0, 0]), array([1, 1, 1])).
            radius (Tuple[float, float], optional): Valid range of radii. Defaults to (0.1, 2.0).
            height (Tuple[float, float], optional): Valid range of heights. Defaults to (0.1, 4.0).
            color (ndarray, optional): Color of cylinder in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            name (str, optional): Name of cylinder, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, name)
        self.radius = radius
        self.height = height