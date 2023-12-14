from typing import Optional, Tuple, Sequence, List, Union
import numpy as np
from abc import ABC, abstractmethod
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import FixedCuboid, FixedSphere, FixedCylinder
from numpy import random, linalg, ndarray


class IsaacObstacle(ABC):
    """ An general obstacle for the isaac environment """
    def __init__(
            self, 
            position: [Union[ndarray, Tuple[ndarray, ndarray]]],
            orientation: [Union[ndarray, Tuple[ndarray, ndarray]]],
            scale: [Union[ndarray, Tuple[ndarray, ndarray]]],
            static: bool,
            velocity: Optional[float] = None,
            length: Optional[Union[float, Tuple[float, float]]] = None,
            direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = None,
            step_count: Optional[float] = None,
            step_size: Optional[float] = None,

    ) -> None:
        self._initPos = position        # save initial pos/ min-max range
        self._initOri = orientation     # save initial ori/ min-max range
        self._initScale = scale         # save initial scale/ min-max range
        
        self.static = static            # save if the obstacle needs a trajecotry
        self._initVelocity = velocity   # save initial velocity/ min-max range
        self._initLength = length       # save initial length/ min-max range
        self._initDirection = direction # save initial direction/ min-max range
        self.step_count = step_count    # needed to calculate step for each update
        self.step_size = step_size      # needed to calculate step for each update

        # if necessary create random position, orientation and scale from range
        self.position = self._getPosition()
        self.orientation = self._getOrientation()
        self.scale = self._getScale()

        if not self.static:
            # create values for a random trajectory the obstacle moves along
            self.velocity = self._getVelocity()
            self.direction = self._getDirection()
            self.length = self._getLength()
            self.goal = self._getGoal()
            self.step = self._getStep()

    # create random position if there is a range given as argument
    def _getPosition(self) -> np.array:
        if isinstance(self._initPos, tuple):
            min, max = self._initPos
            return random.uniform(low=min, high=max, size=(3,))
        else:
            return self._initPos
    
    # create random orientation if there is a range given as argument
    def _getOrientation(self) -> np.array:
        if isinstance(self._initOri, tuple):
            min, max = self._initOri
            return random.uniform(low=min, high=max, size=(4,))
        else:
            return self._initOri
    
    # create random scale if there is a range given as argument
    def _getScale(self) -> np.array:
        if isinstance(self._initScale, tuple):
            min, max = self._initScale
            return random.uniform(low=min, high=max, size=(3,))
        else:
            return self._initScale
    
    # create a random velocity if there is a range given as argument
    def _getVelocity(self):
        if isinstance(self._initVelocity, tuple):
            min, max = self._initVelocity
            return random.uniform(low=min, high=max)
        else:
            return self._initVelocity
        
    # create a random direction if there is a range given as argument
    def _getDirection(self):
        if isinstance(self._initDirection, tuple):
            min, max = self._initDirection
            return random.uniform(low=min, high=max, size=(3,))
        else:
            return self._initDirection
  
    # create a random length if there is a range given as argument
    def _getLength(self):
        if isinstance(self._initLength, tuple):
            min, max = self._initLength
            return random.uniform(low=min, high=max)
        else:
            return self._initLength
    
    # create a goal for the direction and length of the trajactory
    def _getGoal(self):
        return self.position + (self.direction * self.length / linalg.norm(self.direction))
    
    # create a random step the obstacle moves towards the goal on an update
    def _getStep(self):
        return self.velocity * self.step_count * self.step_size
    
    # reset obstacle
    def post_reset(self) -> None:
        # generate new random position, orientation and scale (if necessary)
        self.position = self._getPosition()
        self.orientation = self._getOrientation()
        self.scale = self._getScale()
         
        # apply new values
        self.set_local_scale(self.scale)
        self.set_world_pose(self.position, self.orientation)

        # only for dynamic objects
        if not self.static:
            self.velocity = self._getVelocity()
            self.direction = self._getDirection()
            self.length = self._getLength()
            self.goal = self._getGoal()
            self.step = self._getStep()

    def update(self) -> bool:
        if self.static: return True
        
        # move towards goal
        diff = self.goal - self.position
        diff_norm = linalg.norm(diff)

        if diff_norm > 1e-3:
            step = self.step if diff_norm > self.step else diff_norm  # ensures that we don't jump over the target destination
            step = diff * (step / diff_norm)
            self.velocity = step / (self.step_size * self.step_count)
            self.position = self.position + step   
            self.set_world_pose(self.position, self.orientation)     
        
        return True


class IsaacCube(IsaacObstacle, FixedCuboid):
    def __init__(
        self,
        prim_path: str,
        position: [Union[ndarray, Tuple[ndarray, ndarray]]],
        orientation: [Union[ndarray, Tuple[ndarray, ndarray]]],
        scale: [Union[ndarray, Tuple[ndarray, ndarray]]],
        static: bool,
        name: str = "random_cube",
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        velocity: Optional[Union[float, Tuple[float, float]]] = None,
        length: Optional[Union[float, Tuple[float, float]]] = None,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = None,
        step_count: Optional[float] = None,
        step_size: Optional[float] = None,

    ) -> None:
        
        # Init class that hanldes random values
        IsaacObstacle.__init__(self, position, orientation, scale, static, velocity, length, direction, step_count, 
            step_size) 
        
        # init base class from isaac for a cube 
        FixedCuboid.__init__(self, prim_path=prim_path, name=name, position=self.position, orientation=self.orientation,
            scale=self.scale, color=color, size=size)


class IsaacSphere(IsaacObstacle, FixedSphere):
    def __init__(
        self,
        prim_path: str,
        position: [Union[ndarray, Tuple[ndarray, ndarray]]],
        orientation: [Union[ndarray, Tuple[ndarray, ndarray]]],
        scale: [Union[ndarray, Tuple[ndarray, ndarray]]],
        static: bool,
        name: str = "random_cube",
        color: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        velocity: Optional[Union[float, Tuple[float, float]]] = None,
        length: Optional[Union[float, Tuple[float, float]]] = None,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = None,
        step_count: Optional[float] = None,
        step_size: Optional[float] = None,

    ) -> None:
        
        # Init class that hanldes random values
        IsaacObstacle.__init__(self, position, orientation, scale, static, velocity, length, direction, step_count, 
            step_size) 
        
        # init base class from isaac for a cube 
        FixedSphere.__init__(self, prim_path=prim_path, name=name, position=self.position, orientation=self.orientation,
            scale=self.scale, color=color, radius=radius)


class IsaacCylinder(IsaacObstacle, FixedCylinder):
    def __init__(
        self,
        prim_path: str,
        position: [Union[ndarray, Tuple[ndarray, ndarray]]],
        orientation: [Union[ndarray, Tuple[ndarray, ndarray]]],
        scale: [Union[ndarray, Tuple[ndarray, ndarray]]],
        static: bool,
        name: str = "random_cube",
        color: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        velocity: Optional[Union[float, Tuple[float, float]]] = None,
        length: Optional[Union[float, Tuple[float, float]]] = None,
        direction: Optional[Union[ndarray, Tuple[ndarray, ndarray]]] = None,
        step_count: Optional[float] = None,
        step_size: Optional[float] = None,

    ) -> None:
        
        # Init class that hanldes random values
        IsaacObstacle.__init__(self, position, orientation, scale, static, velocity, length, direction, step_count, 
            step_size) 
        
        # init base class from isaac for a cube 
        FixedCylinder.__init__(self, prim_path=prim_path, name=name, position=self.position, orientation=self.orientation,
            scale=self.scale, color=color, radius=radius, height=height)
                