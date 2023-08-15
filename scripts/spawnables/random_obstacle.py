from typing import List, Tuple, Dict
from numpy import ndarray, array, random
from scripts.spawnables.obstacle import Obstacle

def _get_value_in_range(min: float, max: float, range: float) -> float:
    """Returns a value between min and max.
        Example: Min=0, Max=10, range:0.5 -> 5

    Args:
        min (float): Min value
        max (float): Max value
        range (float): [0, 1]

    Returns:
        float: _description_
    """

    return min + (max - min) * range


class RandomObstacle(Obstacle):
    def __init__(self, position: ndarray, color: List[float], collision: bool, observable: bool, static: bool, name: str = None) -> None:
        super().__init__(position, color, collision, observable, static, name)

class RandomCube(RandomObstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        orientation: Tuple[ndarray, ndarray] = (array([1, 0, 0, 0]), array([0, 0, 1, 0])),
        scale: Tuple[ndarray, ndarray] = (array([0.5, 0.5, 0.5]), array([1.5, 1.5, 1.5])),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
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
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of cube, allowing it to be referenced. Defaults to None.
        """

        # per default, all random obstacles are not-static
        super().__init__(position, color, collision, observable, static, name)

        self.orientation = orientation        
        self.scale = scale
    
    def get_random_state(self) -> Tuple[ndarray, ndarray, ndarray]:
        # generate random numbers to allow setting random properties efficiently
        rand_floats = random.random_sample(3)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(self.position[0], self.position[1], rand_floats[0])
        ori = _get_value_in_range(self.orientation[0], self.orientation[1], rand_floats[1])
        scale = _get_value_in_range(self.scale[0], self.scale[1], rand_floats[2])

        return pos, ori, scale

class RandomSphere(RandomObstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        radius: Tuple[float, float] = (0.1, 2.0),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        name: str=None
    ) -> None:
        """A sphere obstacle spawned in each environment. Each reset, the spheres properties change, as defined by the range specified here.

        Args:
            position (Tuple[ndarray, ndarray], optional): Valid range of positions. Defaults to (array([0, 0, 0]), array([1, 1, 1])).
            radius (Tuple[float, float], optional): Valid range of radii. Defaults to (0.1, 2.0).
            color (ndarray, optional): Color of sphere in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flar for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of sphere, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, static, name)
        self.radius = radius

    def get_random_state(self) -> Tuple[ndarray, ndarray]:
        # generate random numbers to allow setting random properties efficiently
        rand_floats = random.random_sample(2)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(self.position[0], self.position[1], rand_floats[0])
        rad = _get_value_in_range(self.radius[0], self.radius[1], rand_floats[1])

        return pos, rad

class RandomCylinder(RandomObstacle):
    def __init__(
        self,
        position: Tuple[ndarray, ndarray] = (array([0, 0, 0]), array([1, 1, 1])),
        radius: Tuple[float, float] = (0.1, 2.0),
        height: Tuple[float, float] = (0.1, 4.0),
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
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
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of cylinder, allowing it to be referenced. Defaults to None.
        """

        super().__init__(position, color, collision, observable, static, name)
        self.radius = radius
        self.height = height

    def get_random_state(self) -> Tuple[ndarray, ndarray, ndarray]:
        # generate random numbers to allow setting random properties efficiently
        rand_floats = random.random_sample(3)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(self.position[0], self.position[1], rand_floats[0])
        rad = _get_value_in_range(self.radius[0], self.radius[1], rand_floats[1])
        height = _get_value_in_range(self.height[0], self.height[1], rand_floats[2])

        return pos, rad, height