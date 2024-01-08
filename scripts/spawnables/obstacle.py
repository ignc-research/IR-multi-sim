from typing import Optional, List, Tuple, Union, DefaultDict
from numpy import ndarray, array
from scripts.spawnables.spawnable import Spawnable
from abc import abstractmethod

class Obstacle(Spawnable):
    def __init__(self, 
                 position: Union[ndarray, Tuple[ndarray, ndarray]],
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]],
                 scale: Union[ndarray, Tuple[ndarray, ndarray]],
                 color: List[float], 
                 collision: bool, 
                 observable: bool, 
                 static: bool, 
                 velocity: float,
                 name: str=None,
                 endpoint: Union[ndarray, Tuple[ndarray, ndarray]]=None
                ) -> None:
        
        super().__init__(position, orientation, collision, observable, name)

        # parse color
        if isinstance(color, List):
            self.color = array(color)
        else:
            self.color = color

        self.static = static
        self.endpoint = endpoint
        self.velocity = velocity
        self.scale = scale

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
        name: str = None,
        endpoint: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.5, 0.5, 0.5])
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
        super().__init__(position, orientation, scale, color, collision, observable, static, velocity, name, endpoint)

        self.static = static

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, "color":self.color, 
                    "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale": self.scale, "color": self.color, 
                    "velocity": self.velocity, "static": self.static, "endpoint": self.endpoint}
    

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
        name: str=None,
        endpoint: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.5, 0.5, 0.5])
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
        super().__init__(position, orientation, scale, color, collision, observable, static, velocity, name, endpoint)
        
        self.radius = radius
        self.static = static

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "color":self.color,  "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "color":self.color, "velocity": self.velocity, "static":self.static, "endpoint":self.endpoint}
        

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
        name: str=None,
        endpoint: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.5, 0.5, 0.5])
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

        super().__init__(position, orientation, scale, color, collision, observable, static, velocity, name, endpoint)
        
        self.radius = radius
        self.height = height
        self.static = static

    def get_constructor_params(self):
        if self.static:
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "height": self.height*10, "color":self.color,  "static":self.static}
        else: 
            return {"position": self.position, "orientation": self.orientation, "scale":self.scale, 
                    "radius": self.radius*10, "height": self.height*10, "color":self.color, "velocity": self.velocity, 
                    "static": self.static, "endpoint": self.endpoint}
