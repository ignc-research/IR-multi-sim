from typing import List, Tuple, Union
from abc import ABC
from numpy import array, ndarray, random, append
import pybullet as pyb

_spawnedTables = 0

class PyUrdf(ABC):
    """ An urdf specific for the pybullet environment """
    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path
        

class PyTable(PyUrdf):
    """ A table specific for the pybullet environment """
    def __init__(self,
                urdf_path: str,  
                name: str = None, 
                offset: ndarray = array([0,0,0]), 
                position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0,0,0]), 
                orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.,0.,0.,1.]), 
                collision: bool = False, 
                observable: bool = False, 
                static: bool = False, 
                scale: Union[float, Tuple[float, float]] = 1.0,
                limit_lower: ndarray = array([-0.65,-0.4,0.63]),
                limit_upper: ndarray = array([0.65,0.4,0.63]),
            ) -> None:
        
        super().__init__(urdf_path)
        
        # set default name
        if name is None:
            global _spawnedTables
            name = f"table_{_spawnedTables+1}"
            _spawnedTables += 1     
        self.name= name

        self.offset = offset
        self.collision = collision
        self.observable = observable
        self.position = position
        self.orientation = orientation.tolist()
        self.static = static
        self.scale = scale
        self.limit_lower = limit_lower
        self.limit_upper = limit_upper

        # create pybullet instance of table
        self.id = pyb.loadURDF(urdf_path, basePosition=(self.position + self.offset).tolist(), baseOrientation=self.orientation, 
                               useFixedBase=True, globalScaling=self.scale)
        
        def getValidTablePos(self) -> List[float]:
            min = self.limit_lower + self.position + self.offset
            max = self.limit_upper + self.position + self.offset
            return random.uniform(min, max).tolist()
        
        def moveTo(self, pos=None, rot=None) -> None:
            if pos:
                self.position = pos
            if rot:
                self.orientation = rot
            pyb.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self.position, ornObj=self.orientation) 

        def remove(self) -> None:
            pyb.removeBody(self.id) 
