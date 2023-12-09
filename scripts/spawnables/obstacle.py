from typing import Optional, List, Tuple, Union, DefaultDict
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable
from abc import abstractmethod

class Obstacle(Spawnable):
    def __init__(self, position: Union[ndarray, Tuple[ndarray, ndarray]], color: List[float], collision: bool, observable:bool, static:bool, name: str=None) -> None:
        super().__init__(position, color, collision, observable, name)
        self.static = static

    def has_random_position(self):
        return type(self.position) is tuple

    @abstractmethod
    def is_randomized(self):
        pass

    @abstractmethod
    def get_constructor_params(self) -> DefaultDict:
        pass

class Cube(Obstacle):
    def __init__(
        self,
        position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]), 
        orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([1, 0, 0, 0]),
        scale: Union[List[float], Tuple[List[float], List[float]]] = [.1, .1, .1],
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        velocity: Optional[Union[float, Tuple[float, float]]] = 1.,
        length: Optional[Union[float, Tuple[float, float]]] = 1.,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = array([1, 0, 0]),
        name: str = None
    )-> None:
        """A cube obstacle spawned in each environment with fixed parameters.

        Args:
            position (ndarray, optional): Position of cube. Defaults to array([0, 0, 0])
            orientation (ndarray, optional): Orientation of cube in quaternion. Defaults to array([1, 0, 0, 0]).
            scale (List[float], optional): Scale of cube. Defaults to [1., 1., 1.].
            color (ndarray, optional): Color of cube in RGB. Defaults to array([1., 1., 1.]).
            collision (bool, optional): Flag for enabeling collision. Defaults to True.
            observable (bool, optional): Flag for enabeling observability by ML model. Defaults to True.
            static (bool, optional): Flag disabeling gravity to affect the obstacle. Default to True.
            name (str, optional): Name of cube, allowing it to be referenced. Defaults to None.
        """
        super().__init__(position, color, collision, observable, static, name)

        self.scale = scale
        self.static = static
        self.velocity = velocity
        self.length = length
        self.direction = direction

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = array(orientation)
        else:
            self.orientation = orientation

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, "color":self.color, 
                    "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, "color":self.color, 
                    "velocity": self.velocity, "length": self.length, "direction":self.direction, "static":self.static}
    
    def has_random_orientation(self):
        return type(self.orientation) is tuple
    
    def has_random_scale(self):
        return type(self.scale) is tuple
    
    def is_randomized(self):
        return self.has_random_position() or self.has_random_orientation() or self.has_random_scale()

class Sphere(Obstacle):
    def __init__(
        self,
        position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]),
        orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([1, 0, 0, 0]),
        scale: Union[List[float], Tuple[List[float], List[float]]] = [.1, .1, .1],
        radius: Union[float, Tuple[float, float]] = .1,
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        velocity: Optional[Union[float, Tuple[float, float]]] = 1.,
        length: Optional[Union[float, Tuple[float, float]]] = 1.,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = array([1, 0, 0]),
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
        
        self.scale = scale
        self.radius = radius
        self.static = static
        self.velocity = velocity
        self.length = length
        self.direction = direction

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = array(orientation)
        else:
            self.orientation = orientation

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "color":self.color,  "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "color":self.color, "velocity": self.velocity, "length": self.length, 
                    "direction":self.direction, "static":self.static}

    def has_random_radius(self):
        return type(self.radius) is tuple
    
    def is_randomized(self):
        return self.has_random_position() or self.has_random_radius()

class Cylinder(Obstacle):
    def __init__(
        self,
        position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]),
        orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([1, 0, 0, 0]),
        scale: Union[List[float], Tuple[List[float], List[float]]] = [.1, .1, .1], 
        radius: Union[float, Tuple[float, float]] = .1,
        height: Union[float, Tuple[float, float]] = .1,
        color: ndarray = array([1., 1., 1.]),
        collision: bool = True,
        observable: bool = True,
        static: bool = True,
        velocity: Optional[Union[float, Tuple[float, float]]] = 1.,
        length: Optional[Union[float, Tuple[float, float]]] = 1.,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = array([1, 0, 0]),
        name: str=None,
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
        self.scale = scale
        self.static = static
        self.velocity = velocity
        self.length = length
        self.direction = direction

        # parse orientation
        if isinstance(orientation, List):
            self.orientation = array(orientation)
        else:
            self.orientation = orientation

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "height": self.height*10, "color":self.color,  "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "height": self.height*10, "color":self.color, "velocity": self.velocity, 
                    "length": self.length, "direction":self.direction, "static":self.static}
    
    def has_random_radius(self):
        return type(self.radius) is tuple
    
    def has_random_height(self):
        return type(self.height) is tuple

    def is_randomized(self):
        return self.has_random_position() or  self.has_random_radius() or self.has_random_height()