from typing import List, Tuple, Union
from abc import ABC
from numpy import array, ndarray, random, append
import numpy as np
import pybullet as pyb

_spawnedRobots = 0

class PyRobot(ABC):
    """ A robot specific for the pybullet environment """
    def __init__(self,
                 urdf_path: str, control_type: str, max_velocity: float,
                 observableJoints: List[str], 
                 name: str = None, 
                 offset: ndarray = array([0,0,0]), 
                 position: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.,0.,0.]), 
                 orientation: Union[ndarray, Tuple[ndarray, ndarray]] = array([0.,0.,0.,1.]), 
                 collision: bool = True, 
                 observable: bool = True, 
                 scale: float = 1.0,
                ) -> None:
        
        # set default name
        if name is None:
            global _spawnedRobots
            name = f"robot_{_spawnedRobots+1}"
            _spawnedRobots += 1     
        self.name= name

        self.scale = scale    
        self.offset = offset
        self.collision = collision
        self.observable = observable
        self.urdf_path = urdf_path
        self.control_type = control_type
       
        # save initial position and orientation
        self._initPos = position
        self._initOri = orientation

        # for random joint configuration
        self.randomJoints = True

        # if necessary create random position and orientation
        self.position = self._getPosition()
        self.orientation = self._getOrientation()

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
        self.maxVelocity = []
        self.maxForce = []
        self.lower = np.array([])
        self.upper = np.array([])

        for i in range(pyb.getNumJoints(self.id)):
            info = pyb.getJointInfo(self.id, i) 

            jointName = info[12].decode('UTF-8')                # string name of joint
            controllable = False if info[2] == 4 else True      # 4 == fixed joint
            lower, upper = info[8:10]                           # limits (min,max) of each controllable joint
            jointAngle = pyb.getJointState(self.id, i)[0]       # initial angle for resets
            maxVel = info[11]
            maxF = info[10]

            self.jointIds.append(i)
            self.jointNames.append(name + "/" + jointName)
            self.initialJoints.append(jointAngle)
                 
            if controllable:
                self.controllableJoints.append(i) 
                self.lower = np.append(self.lower, lower)
                self.upper = np.append(self.upper, upper)
                self.limits.append((lower,upper))
                self.maxForce.append(maxF)
                
                if max_velocity:
                    # use max vel from yaml config
                    self.maxVelocity.append(max_velocity) 
                else:
                    # max vel from robot urdf
                    self.maxVelocity.append(maxVel)   
                
            if jointName in observableJoints:
                self.observableJoints.append(i)
                self.observableJointNames.append(jointName)

        # if robot has random position, move joints into random config
        if self.randomJoints:
            valid = False
            while not valid:
                tmpPos = self._getJointPositions()
                for i, joint in enumerate(self.controllableJoints):
                    pyb.resetJointState(bodyUniqueId=self.id, jointIndex=joint, targetValue=tmpPos[i]) 
                    valid = self._checkValidJointConfig()
       
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
    
    # return angles of all controllable robot joints
    def getJointAngles(self) -> List[float]:
        angles = array([])
    
        for joint in self.controllableJoints:
            angle = pyb.getJointState(self.id, joint)[0] 
            angles = append(angles, angle)
        
        return angles
    
    # deletes and recreates robot with new random config
    def reset(self) -> None:
        #pyb.removeBody(self.id) 
        #self.id = pyb.loadURDF(self.urdf_path, basePosition=self._getPosition(), baseOrientation=self._getOrientation(), 
        #                       useFixedBase=True, globalScaling=self.scale)
        
        # reset robot
        pyb.resetBasePositionAndOrientation(bodyUniqueId=self.id,
                                            posObj=self._getPosition(),
                                            ornObj=self._getOrientation())

        # if robot has random position, move joints into random config
        if self.randomJoints:
            valid = False
            while not valid:
                tmpPos = self._getJointPositions()
                for i, joint in enumerate(self.controllableJoints):
                    pyb.resetJointState(bodyUniqueId=self.id, jointIndex=joint, targetValue=tmpPos[i]) 
                    valid = self._checkValidJointConfig()     

    # create random position if there is a range given as argument
    def _getPosition(self) -> List[float]:       
        if isinstance(self._initPos, tuple):
            min, max = self._initPos
            return (random.uniform(min, max) + self.offset).tolist()
        else:
            return (self._initPos + self.offset).tolist()

    # create random orientation if there is a range given as argument
    def _getOrientation(self) -> List[float]:
        if isinstance(self._initOri, tuple):
            min, max = self._initOri
            return (random.uniform(min, max)).tolist()
        else:
            return (self._initOri).tolist()

    # move all joints into a random configuration if robot as a random position
    def _getJointPositions(self) -> List[float]:
        if self.randomJoints:
            return [random.uniform(limit[0], limit[1]) for limit in self.limits]
        else: 
            return self.initialJoints
        
    def _checkValidJointConfig(self) -> bool:
        return True
    
        # TODO
        pyb.performCollisionDetection() 
        contactPoints = pyb.getContactPoints()  # get collisions
        if not contactPoints: return True # skip if there are no collisions

        # extract all colisions
        collisions = [] 
        for point in contactPoints:
            # pyb may have contacts with separation dist greater zero    
            if point[8] <= 0: 
                collisions.append((point[1], point[2]))

        # report collisions
        finalCollisions = [tup for tup in collisions if not any(val == 0 for val in tup)]
        if finalCollisions: 
            return False
        return True


   
