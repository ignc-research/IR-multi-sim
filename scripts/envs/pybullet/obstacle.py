from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from numpy import array, ndarray, random
import pybullet as pyb

_spawnedCubes = 0
_spawnedSpheres = 0
_spawnedCylinders = 0

class PyObstacle(ABC):
    """ An obstacle specific for the pybullet environment """
    def __init__(self, name: str, 
                 position: ndarray, 
                 offset: ndarray, 
                 orientation: ndarray, 
                 static: bool, 
                 collision: bool, 
                 color: ndarray
                ) -> None:
        
        self.offset = offset
        self.static = static
        self.collision = collision
        self.color = color

        # save initial positionand orientation
        self._initPos = position
        self._initOri = orientation

        # if necessary create random position and orientation else set as defined
        self.position = self._getPosition()
        self.orientation = self._getOrientation()


    # create random position if there is a range given as argument
    def _getPosition(self) -> List[float]:
        if isinstance(self._initPos, tuple):
            self.randomJoints = True
            min, max = self._initPos
            return (random.uniform(min, max) + self.offset).tolist()
        else:
            self.randomJoints = False
            return (self._initPos + self.offset).tolist()

    # create random orientation if there is a range given as argument
    def _getOrientation(self) -> List[float]:
        if isinstance(self._initOri, tuple):
            min, max = self._initOri
            return random.uniform(min, max).tolist()
        else:
            return self._initOri.tolist()
    
    @abstractmethod
    def reset():
        pass

    @abstractmethod
    def getPose():
        pass
    
                
class PyCube(PyObstacle):
    """A cube obstacle spawned in each environment with fixed parameters. """
    def __init__(self, 
                 name: str = None, 
                 offset: ndarray = array([0,0,0]), 
                 position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]), 
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0., 0., 0., 1.]), 
                 scale: Union[List[float], Tuple[List[float], List[float]]] = array([1, 1, 1]), 
                 static: bool = True, 
                 collision: bool = False, 
                 color: ndarray = array([1, 1, 1])             
                ) -> None:
            
        super().__init__(name, position, offset, orientation, static, collision, color)

        self._initScale = scale
        self.scale = self._getScale()

        # set default name
        if name.startswith("obj"):
            global _spawnedCubes
            name = f"cube_{_spawnedCubes+1}"
            _spawnedCubes += 1     
        self.name= name
       
        # create pybullet object 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX,halfExtents=[x/2 for x in self.scale], rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in self.scale]) if self.collision else -1,
            basePosition=self.position,
            baseOrientation=self.orientation
        )

    # create random scale if there is a range given as argument
    def _getScale(self) -> float:
        if isinstance(self._initScale, tuple):
            min, max = self._initScale
            return random.uniform(min, max)
        else:
            return self._initScale

    def getPose(self) -> Tuple[List[float], List[float], List[float]]:
        pos, rot = pyb.getBasePositionAndOrientation(self.id) 
        pos -= self.offset
        return pos, rot, tuple(self.scale)
    
    def reset(self) -> None:
        pyb.removeBody(self.id) 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX,halfExtents=[x/2 for x in self._getScale()], rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in self._getScale()]) if self.collision else -1,
            basePosition=self._getPosition(),
            baseOrientation=self._getOrientation()
        )    
    

class PySphere(PyObstacle):
    """A sphere obstacle spawned in each environment with fixed parameters. """
    def __init__(self, 
                 name: str = None, 
                 offset: ndarray = array([0,0,0]), 
                 position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]), 
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0., 0., 0., 1.]), 
                 radius: Union[float, Tuple[float, float]] = 1.,
                 static: bool = True, 
                 collision: bool = False, 
                 color: ndarray = array([1, 1, 1])             
                ) -> None:
    
        super().__init__(name, position, offset, orientation, static, collision, color)

        self._initRadius = radius
        self.radius = self._getRadius()

        # set default name
        if name.startswith("obj"):
            global _spawnedSpheres
            name = f"sphere_{_spawnedSpheres+1}"
            _spawnedSpheres += 1     
        self.name= name
       
        # create pybullet object 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.radius, rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=self.radius) if self.collision else -1,
            basePosition=self.position,
            baseOrientation=self.orientation
        )

    # create random scale if there is a range given as argument
    def _getRadius(self) -> float:
        if isinstance(self._initRadius, tuple):
            min, max = self._initRadius
            return random.uniform(min, max)
        else:
            return self._initRadius
    
    def getPose(self) -> Tuple[List[float], List[float], List[float]]:
        pos, rot = pyb.getBasePositionAndOrientation(self.id) 
        pos -= self.offset
        return pos, rot, tuple((self.radius,0,0))
    
    def reset(self) -> None:
        pyb.removeBody(self.id) 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self._getRadius(), rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=self._getRadius()) if self.collision else -1,
            basePosition=self._getPosition(),
            baseOrientation=self._getOrientation()
        ) 
    

class PyCylinder(PyObstacle):
    """A cylinder obstacle spawned in each environment with fixed parameters. """
    def __init__(self, 
                 name: str = None, 
                 offset: ndarray = array([0,0,0]), 
                 position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0, 0, 0]), 
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0., 0., 0., 1.]), 
                 radius: Union[float, Tuple[float, float]] = 1.,    # radius of cylinder
                 height: Union[float, Tuple[float, float]] = 1.,
                 static: bool = True, 
                 collision: bool = False, 
                 color: ndarray = array([1, 1, 1])             
                ) -> None:
    
        super().__init__(name, position, offset, orientation, static, collision, color)

        # set default name
        if name.startswith("obj"):
            global _spawnedCylinders
            name = f"cylinder_{_spawnedCylinders+1}"
            _spawnedCylinders += 1     
        self.name= name

        self._initRadius = radius
        self._initHeight = height
        self.radius = self._getRadius()
        self.height = self._getHeight()

        # create pybullet object 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=self.radius, length=self.height, rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=self.radius, height=self.height) if self.collision else -1, 
            basePosition=self.position,
            baseOrientation=self.orientation
        )

    # create random scale if there is a range given as argument
    def _getRadius(self) -> float:
        if isinstance(self._initRadius, tuple):
            min, max = self._initRadius
            return random.uniform(min, max)
        else:
            return self._initRadius
        
    # create random height if there is a range given as argument
    def _getHeight(self) -> float:
        if isinstance(self._initHeight, tuple):
            min, max = self._initHeight
            return random.uniform(min, max)
        else:
            return self._initHeight   
    
    # return the position of the object
    def getPose(self) -> Tuple[List[float], List[float], List[float]]:
        pos, rot = pyb.getBasePositionAndOrientation(self.id) 
        pos -= self.offset
        return pos, rot, tuple((self.height,self.height,0))

    def reset(self) -> None:
        pyb.removeBody(self.id) 
        self.id = pyb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=self._getRadius(), 
                                                       length=self._getHeight(), rgbaColor=self.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=self._getRadius(), 
                                                             height=self._getHeight()) if self.collision else -1, 
            basePosition=self._getPosition(),
            baseOrientation=self._getOrientation()
        ) 