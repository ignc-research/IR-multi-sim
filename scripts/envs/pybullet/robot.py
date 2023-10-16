from typing import List, Tuple, Union
from abc import ABC
from numpy import array, ndarray, random, append
import pybullet as pyb

_spawnedRobots = 0

class PyRobot(ABC):
    """ An robot specific for the pybullet environment """
    def __init__(self,
                 urdf_path: str, 
                 observableJoints: List[str], 
                 name: str = None, 
                 offset: ndarray = array([0,0,0]), 
                 position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0,0,0]), 
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.,0.,0.,1.]), 
                 collision: bool = True, 
                 observable: bool = True, 
                 scale: Union[float, Tuple[float, float]] = 1.0,
                ) -> None:
        
        # set default name
        if name is None:
            global _spawnedRobots
            name = f"robot_{_spawnedRobots+1}"
            _spawnedRobots += 1     
        self.name= name

        self.offset = offset
        self.collision = collision
        self.observable = observable

        # save initial position, orientation and scale
        self._initPos = position
        self._initOri = orientation
        self._initScale = scale
        
        # if necessary create random position, orientation and scale else set as defined
        self.position = self._getPosition()
        self.orientation = self._getOrientation()
        self.scale = self._getScale()

        # create pybullet instance of robot
        self.id = pyb.loadURDF(urdf_path, basePosition=self.position, baseOrientation=self.orientation, 
                               useFixedBase=True, globalScaling=self.scale)
        
        # track robot joints
        self.jointIds = []
        self.jointNames = []
        self.controllableJoints = []
        self.observableJoints = []
        self.observableJointNames = []
        self.limits = []
        self.initialJoints = []
                    
        for i in range(pyb.getNumJoints(self.id)):
            info = pyb.getJointInfo(self.id, i) 

            jointName = info[12].decode('UTF-8')                # string name of the joint
            controllable = info[2]                              # moveable joints for executing a action
            lower, upper = info[8:10]                           # limits (min,max) of each controllable joint
            jointAngle = pyb.getJointState(self.id, i)[0]       # initial angle for resets

            self.limits.append((lower,upper))
            self.jointIds.append(i)
            self.jointNames.append(name + "/" + jointName)
            self.initialJoints.append(jointAngle)
            
            if controllable != 4:
                self.controllableJoints.append(i)       # no fixed joint
                
            
            if jointName in observableJoints:
                self.observableJoints.append(i)
                self.observableJointNames.append(jointName)

        # if robot has random position, move joints into random config
        tmpPos = self._getJointPositions()
        for i, joint in enumerate(self.jointIds):
            pyb.resetJointState(bodyUniqueId=self.id, jointIndex=joint, targetValue=tmpPos[i]) 
       
    # returns current robot position, orientation and scale
    def getPose(self) -> Tuple[List[float], List[float], float]:
        pos, rot = pyb.getBasePositionAndOrientation(self.id) 
        pos -= self.offset
        return tuple(pos), rot, ((self.scale,self.scale,self.scale))

    # return the position and rotation of all observable robot joints
    def getObservableJointsPose(self) -> Tuple[List[float], List[float]]:
        positions = array([])
        rotations = array([])
    
        for joint in self.observableJoints:
            pos, rot = pyb.getLinkState(self.id, joint)[:2]
            pos -= self.offset
            
            positions = append(positions, pos)
            rotations = append(rotations, rot)
            
        return positions, rotations

    # creates new random position for the robot joints
    def reset(self) -> None:
        # reset robot
        pyb.resetBasePositionAndOrientation(bodyUniqueId=self.id,
                                            posObj=self._getPosition(),
                                            ornObj=self._getOrientation())
        
        # reset robot joints
        tmpPos = self._getJointPositions()
        for i, joint in enumerate(self.jointIds):
            pyb.resetJointState(bodyUniqueId=self.id, jointIndex=joint, targetValue=tmpPos[i]) 
        
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
            return (random.uniform(min, max)).tolist()
        else:
            return (self._initOri).tolist()

    # create random scale if there is a range given as argument
    def _getScale(self) -> float:
        if isinstance(self._initScale, tuple):
            min, max = self._initScale
            return (min + max) * 0.5  # cannot change scale during runtime
        else:
            return self._initScale

    # move all joints into a random configuration if robot as a random position
    def _getJointPositions(self) -> List[float]:
        if self.randomJoints:
            return [random.uniform(limit[0], limit[1]) for limit in self.limits]
        else: 
            return self.initialJoints

   
