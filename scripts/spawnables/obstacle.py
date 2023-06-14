from typing import List, Dict
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable

class Obstacle(Spawnable):
    def __init__(self, position: ndarray, color: List[float], collision: bool, observable:bool, name: str=None) -> None:
        super().__init__(position, color, collision, observable, name)


class Cube(Obstacle):
    def __init__(
        self,
        position: ndarray = array([0, 0, 0]), 
        orientation: ndarray = array([1, 0, 0, 0]),
        scale: List[float] = [1., 1., 1.],
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        name: str=None
    )-> None:
        super().__init__(position, color, collision, observable, name)
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
        name: str=None
    ) -> None:
        super().__init__(position, color, collision, observable, name)
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
        name: str=None
    ) -> None:
        super().__init__(position, color, collision, observable, name)
        self.radius = radius
        self.height = height


def parse_obstacle(params: dict) -> Obstacle:
    selector = {
        "Cube" : Cube,
        "Sphere" : Sphere,
        "Cylinder" : Cylinder
    }

    # extract required type
    type = params["type"]

    # make sure parsing of obstacle type is implemented
    if type not in selector:
        raise Exception(f"Obstacle parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed obstacle
    return selector[type](**params)