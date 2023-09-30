from scripts.envs.modular_env import ModularEnv
from scripts.envs.params.env_params import EnvParams
from scripts.envs.params.control_type import ControlType

from scripts.spawnables.obstacle import Obstacle, Cube, Sphere, Cylinder

from scripts.spawnables.robot import Robot

from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance, calc_distance
from scripts.rewards.timesteps import ElapsedTimesteps

from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset

from stable_baselines3.common.vec_env.base_vec_env import *
from pathlib import Path
from typing import List, Tuple

import pybullet as pyb
import numpy as np
import math

class PybulletEnv(ModularEnv):
    def __init__(self, params: EnvParams) -> None:

        # setup asset path to allow importing robots
        self.asset_path = str(Path().absolute().joinpath(params.asset_path)) + "/"

        # setup basic information about simulation
        self.num_envs = params.num_envs
        self.robot_count = len(params.robots)
        self.observable_robots_count = len([r for r in params.robots if r.observable])
        self.observable_robot_joint_count = sum(len(r.observable_joints) for r in params.robots)
        self.obstacle_count = len(params.obstacles)
        self.observable_obstacles_count = len([o for o in params.obstacles if o.observable])
        self._timesteps: List[int] = np.zeros(params.num_envs)
        self.step_count = params.step_count
        self.step_size = params.step_size
        self.control_type = params.control_type
        self.verbose = params.verbose
        self.initState = None

        # save the distances in the current environment
        self._distance_funcions = []
        self._distances: Dict[str, np.ndarray] = {}
        self._distances_after_reset: Dict[str, np.ndarray] = {}

        # calculate env offsets
        break_index = math.ceil(math.sqrt(self.num_envs))
        self._env_offsets = dict(zip(
            [i for i in range(params.num_envs)],
            [np.array([(i % break_index) * params.env_offset[0], math.floor(i / break_index) * params.env_offset[1], 0]) for i in range(params.num_envs)]
        ))

        # setup PyBullet simulation environment and interfaces
        disp = pyb.DIRECT if params.headless else pyb.GUI
        pyb.connect(disp)
        pyb.setTimeStep(self.step_size)
        pyb.setPhysicsEngineParameter(numSubSteps=1)
        pyb.setGravity(0, 0, -9.8) 
        pyb.setRealTimeSimulation(0)
        pyb.resetSimulation()
        pyb.loadURDF(self.asset_path  + "workspace/plane.urdf", [0,0,-0.01])    # load ground plane

        ### allow tracking spawned objects ###
        self._robots: List = []               # Tupel: (name, robotId, jointNames, joints, observableJoints, controllableJoint, initialPos)
        self._observable_robots: List = []    # Tupel: (name, robotId, jointNames, joints, observableJoints, controllableJoint, initialPos)
        self._obstacles: List = []              # Tupel: (name:str, obstacleID:int, position:array, rotation:list)
        self._observable_obstacles: List = []   # Tupel: (name:str, obstacleID:int, position:array, rotation:list)

        # setup rl environment
        self._setup_environments(params.robots, params.obstacles)
        self._setup_rewards(params.rewards)
        self._setup_resets(params.rewards, params.resets)

        # init bace class last, allowing it to automatically determine action and observation space
        super().__init__(params)

    
    def _setup_environments(self, robots: List[Robot], obstacles: List[Obstacle]) -> None:
        # spawn objects for each environment
        for env_idx in range(self.num_envs):    
            
            # spawn robots
            for robot in robots:
                self._spawn_robot(robot, env_idx)

            # spawn obstacles
            for obstacle in obstacles:
                if isinstance(obstacle, Cube):
                    self._spawn_cube(obstacle, env_idx)
                elif isinstance(obstacle, Sphere):
                    self._spawn_sphere(obstacle, env_idx)
                elif isinstance(obstacle, Cylinder):
                    self._spawn_cylinder(obstacle, env_idx)
                else:
                    raise f"Obstacle {type(obstacle)} not implemented"
        
        # save start configuration for fast reset
        self.initState = pyb.saveState()


    def _setup_rewards(self, rewards: List[Reward]) -> None:
        self._reward_fns = []

        for reward in rewards:
            if isinstance(reward, Distance):
                self._reward_fns.append(self._parse_distance_reward(reward))
            elif isinstance(reward, ElapsedTimesteps):
                self._reward_fns.append(self._parse_timestep_reward(reward))
            else:
                raise f"Reward {type(reward)} not implemented!"


    def _parse_distance_reward(self, distance: Distance):
        # parse indices in observations
        obj1_start = self._find_observable_object(distance.obj1)
        obj2_start = self._find_observable_object(distance.obj2)

        # extract name to allow created function to access it easily
        name = distance.name    
        distance_weight = distance.distance_weight
        orientation_weight = distance.orientation_weight
        normalize = distance.normalize
        exponent = distance.exponent

        # parse function calculating distance
        def distance_per_env() -> Tuple[str, np.ndarray]:
            # calculate distances as np array
            result = []
            for i in range(self.num_envs):
                # extract x,y,z coordinates of objects
                pos1 = self._obs[i][obj1_start:obj1_start+3]
                pos2 = self._obs[i][obj2_start:obj2_start+3]

                # extract quaternion orientation from objects
                ori1 = self._obs[i][obj1_start+3:obj1_start+7]
                ori2 = self._obs[i][obj2_start+3:obj2_start+7]

                result.append(calc_distance(pos1, pos2, ori1, ori2))
            result = np.array(result)

            print("\nResult:", name, result)

            return name, result
    
        # add to existing distance functions
        self._distance_funcions.append(distance_per_env)

        def calculate_distance_reward() -> float:
            # get current distances
            distance_space, distance_orientation = self._get_distance_and_rotation(name)

            # apply weight factor
            weighted_space = distance_weight * distance_space
            weighted_orientation = orientation_weight * distance_orientation

            # skip normalization
            if not normalize:
                return (weighted_space + weighted_orientation) ** exponent

            # retrieve distences after last reset
            begin_space, begin_orient = self._distances_after_reset[name]

            # calculate variance to previous distance, avoiding division by zero
            normalized_space = np.where(begin_space == 0, weighted_space, distance_weight * distance_space / begin_space) 
            normalized_orient = np.where(begin_orient == 0, weighted_orientation, distance_orientation * distance_orientation / begin_orient) 

            return (normalized_space + normalized_orient) ** exponent

        return calculate_distance_reward
    
    
    def _parse_timestep_reward(self, elapsed: ElapsedTimesteps):
        weight = elapsed.weight

        # reward elapsed timesteps
        def timestep_reward():
            return self._timesteps * weight
        
        return timestep_reward


    def _find_observable_object(self, name: str) -> int:
        """
        Given the name of an observable object, tries to retrieve its index in the observations list.
        Example: When two robots are observable and the position of the second robots is being queried, returns index 1.
        """
        # robots are input first into observations
        for index, robot in enumerate(self._observable_robots):
            if robot[0].endswith(name):               
                # first 3 entries are the pos values, the next 4 rot values
                startPos = index*7
                return startPos
 
        obsRobotsAndJoints = self._getObservableJoints(0)
        for _, joints in obsRobotsAndJoints:
            for index, joint in enumerate(joints):
                if name.endswith(joint[1]):
                    # * 7 because we have 3pos, 4rot entries per robot/joint
                    startPos = index*7 + self.observable_robots_count*7
                    return startPos

        # obstacles third
        for index, obstacle in enumerate(self._observable_obstacles):
            if obstacle[0].endswith(name):
                startPos = index*7 + self.observable_robots_count*7 + self.observable_robot_joint_count*7
                return startPos

        raise f"Object {name} must be observable if used for reward"


    def _setup_resets(self, rewards: List[Reward], resets: List[Reset]):
        # make sure that all resets referencing a distance are valid
        distance_names = [r.name for r in rewards if isinstance(r, Distance)]

        self._reset_fns = []
        for reset in resets:
            if isinstance(reset, DistanceReset):
                # make sure that the referenced distance exists
                assert reset.distance_name in distance_names, f"DistanceReset {reset} references distance {reset.distance}, which doesn't exists in rewards!"
                
                self._reset_fns.append(self._parse_distance_reset(reset))
            elif isinstance(reset, TimestepsReset):
                self._reset_fns.append(self._parse_timesteps_reset(reset))
            else:
                raise f"Reset {type(reset)} not implemented!"


    def _parse_distance_reset(self, reset: DistanceReset):
        # extract name to allot created function to access it easily
        name = reset.distance_name
        distance_min, distance_max = reset.min_distance, reset.max_distance
        max_angle = reset.max_angle

        # parse function
        def reset_condition() -> np.ndarray:
            # get distances and rotation of current timestep
            distance, rotation = self._get_distance_and_rotation(name)

            # return true whenever the distance exceed max or min value
            distance_reset = np.where(distance_min <= distance, np.where(distance <= distance_max, False, True), True)
            rotation_reset = np.where(np.abs(rotation) > max_angle, True, False)

            return np.logical_or(distance_reset, rotation_reset)

        return reset_condition


    def _parse_timesteps_reset(self, reset: TimestepsReset):
        max_value = reset.max

        # parse function
        def reset_condition() -> np.ndarray:
            # return true whenever the current timespets exceed the max value
            return np.where(self._timesteps < max_value, False, True)

        return reset_condition


    def step_async(self, actions: np.ndarray) -> None:
        for envId in range(self.num_envs):
            robots = self._getRobots(envId)
 
            # perform action for each robot in the current environment
            for robot in robots:
                robotId, controllableJoints = robot[1], robot[5]

                if self.control_type == ControlType.Velocity:
                    action = [actions[envId] for i in controllableJoints]
                    pyb.setJointMotorControlArray(bodyUniqueId=robotId, jointIndices=controllableJoints, controlMode=pyb.VELOCITY_CONTROL, targetVelocities=action) 
                
                elif self.control_type == ControlType.Position:
                    action = [actions[envId][i] for i in controllableJoints] 
                    pyb.setJointMotorControlArray(bodyUniqueId=robotId, jointIndices=controllableJoints, controlMode=pyb.POSITION_CONTROL, targetPositions=action) 
                
                else:
                    raise Exception(f"Control type {self.control_type} not implemented!")

            # step simulation amount of times according to params
            for _ in range(self.step_count):
                pyb.stepSimulation()  
            
            if self.headless:
                pyb.performCollisionDetection()  
            
           # get Collisions
            self._on_contact_report_event()

    
    def step_wait(self) -> VecEnvStepReturn:
        self._obs = self._get_observations()    # get observations
        self._distances = self._get_distances() # get distances after observations were updated
        self._rewards = self._get_rewards()     # get rewards
        self._dones = self._get_dones()         # get dones

        #print("Obs    :", self._obs)
        #print("Dist.  :", self._distances)
        #print("Rewards:", self._rewards)
        #print("Dones  :", self._dones)
        #print("Timest.:", self._timesteps)

        # log rewards
        if self.verbose > 0:
            self.set_attr("average_rewards", np.average(self._rewards))

        return self._obs, self._rewards, self._dones, self.env_data


    def reset(self, env_idxs: np.ndarray=None) -> VecEnvObs:
        #pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI,  int(False))

        # reset entire simulation
        if env_idxs is None:
            pyb.restoreState(self.initState)                    # load inital state of env
            self._timesteps = np.zeros(self.num_envs)           # reset timestep tracking
            self._obs = self._get_observations()                # reset observations 
            self._distances_after_reset = self._get_distances() # calculate new distances

            # set dummy value for distances to avoid KeyError
            if self.verbose > 1:
                for name, _ in self._distances_after_reset.items():
                    self.set_attr("distance_"+name, None)   
        
        # reset envs manually
        else:
            self._timesteps[env_idxs] = 0    # reset timestep tracking
           
            # select each environment
            for i in env_idxs:
                
                # reset all obstacles to default pose
                for _, id, pos, rot in self._getObstacles(i):
                    pyb.resetBasePositionAndOrientation(id, pos, rot)
                
                # reset all robot joints to default pose
                for joint in self._getControllableJoints(i):
                    robotId, jointIds, initPos = joint[0], joint[1], joint[2]
                    pos = [[elem] for elem in initPos]      # multiDof function takes list of lists for targetpos
                    pyb.resetJointStatesMultiDof(robotId, jointIds, pos)     


            # log distances of environments which were reset
            if self.verbose > 1:
                for name, distance in self._distances.items():
                    self.set_attr("distance_"+name, np.average(distance[env_idxs]))

            # reset observations # todo: only recalculate necessary observations
            self._obs = self._get_observations()

            # note new distances # todo: only recalculate necessary distances
            self._distances_after_reset = self._get_distances()

        #pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, int(True))

        return self._obs
    

    def get_robot_dof_limits(self) -> List[Tuple[float, float]]:
        limits = [] # init array

        # only get dof limits from robots of first environment
        for robot in self._getRobots(0):
            robotId, joints = robot[1], robot[3]

            for joint in joints:
                lower, upper = pyb.getJointInfo(robotId, joint)[8:10]
                limits.append((lower, upper))
        return limits
    

    def _get_distance_and_rotation(self, name: str) -> Tuple[float, float]:
        distances = self._distances[name]   # get current distances
        # return distance_space (meters), distance_orientation (angle)
        return distances[0], distances[1]
    
    def _getObstacles(self, env_idx: int):
        start_idx = env_idx * self.obstacle_count
        return [self._obstacles[i] for i in range(start_idx, start_idx + self.obstacle_count)]
    
    def _getObservalbeObstacles(self, env_idx: int):
        start_idx = env_idx * self.observable_obstacles_count
        return [self._observable_obstacles[i] for i in range(start_idx, start_idx + self.observable_obstacles_count)]

    def _getRobots(self, env_idx: int):
        start_idx = env_idx * self.robot_count
        return [self._robots[i] for i in range(start_idx, start_idx + self.robot_count)]
 
    def _getObservableRobots(self, env_idx: int):
        start_idx = env_idx * self.observable_robots_count
        return [self._observable_robots[i] for i in range(start_idx, start_idx + self.observable_robots_count)]
 
    def _getJoints(self, env_idx: int):
        # return tupels(robotId,jointIds)
        robots = self._getRobots(env_idx) 
        res = []
        for robot in robots:
           res += (robot[1], robot[3])      
        return res

    def _getObservableJoints(self, env_idx: int):
        # return tupels(robotIds,jointIds)
        robots = self._getRobots(env_idx)          
        res = []
        for robot in robots:
           res.append((robot[1], robot[4]))
        return res
    
    def _getControllableJoints(self, env_idx: int):
        # return tupels(robotId,jointIds, initPos)
        robots = self._getRobots(env_idx)         
        res = []
        for robot in robots:
           robotId, controllJoints = robot[1], robot[5]
           initPos = [robot[6][i] for i in controllJoints]
           res.append((robotId, controllJoints, initPos))
        return res

    def _get_observations(self) -> VecEnvObs:
        obs = []

         # iterate through each environment
        for env_idx in range(self.num_envs):
            env_obs = []
    
            # get observations from all robots in environment
            robots = self._getObservableRobots(env_idx)
            for robot in robots:
                pos, rot = pyb.getBasePositionAndOrientation(robot[1])    # get its pose
                pos -= self._env_offsets[env_idx]                         # apply env offset

                # add robot pos and rotation to list of observations
                env_obs.extend(pos)
                env_obs.extend(rot)

            # get observations from all observable joints in environment
            robotAndJoints = self._getObservableJoints(env_idx)
            for robotId, joints in robotAndJoints:
                for joint in joints:
                    pos, rot = pyb.getLinkState(robotId, joint[0])[:2]     # get its pose 
                    pos -= self._env_offsets[env_idx]                       # apply env offset

                    # add pos and rotation to list of observations
                    env_obs.extend(pos)
                    env_obs.extend(rot)

            # get observations from all obstacles in environment
            obstacles = self._getObservalbeObstacles(env_idx)
            for obstacle in obstacles:
                pos, rot = pyb.getBasePositionAndOrientation(obstacle[1])   # get its pose  
                pos -= self._env_offsets[env_idx]                           # apply env offset        

                # add obstacle pos and rotation to list of observations
                env_obs.extend(pos)
                env_obs.extend(rot)

            # add observations gathered in environment to dictionary
            obs.append(env_obs)
        return np.array(obs)


    def _get_distances(self) -> Dict[str, np.ndarray]:
        distances = {}      # reset current distances

        for distance_fn in self._distance_funcions:
            name, distance = distance_fn()      # calcualte current distances
            distances[name] = distance

        return distances


    def _get_rewards(self) -> List[float]:
        rewards = np.zeros(self.num_envs)
        print("rewards:", rewards)

        for fn in self._reward_fns:
            curr = fn()
            print("Curr", curr)
            rewards += curr

        return rewards


    def _get_dones(self) -> List[bool]:
        dones = np.full(self.num_envs, False)    # init default array: No env is done

        # check if any of the functions specify a reset
        for fn in self._reset_fns:
            dones = np.logical_or(dones, fn())

        self._timesteps = np.where(dones, 0, self._timesteps + 1)   # increment elapsed timesteps if env isn't done
        reset_idx = np.where(dones)[0]                              # reset evns where dones == True

        # reset environments if necessary
        if reset_idx.size > 0:
            self.reset(reset_idx)
        return dones


    def _on_contact_report_event(self) -> None:
        # get collisions
        contactPoints = pyb.getContactPoints() 
        
        # do nothing if there are no collisions
        if len(contactPoints) <= 0:
            return
         
        # extract all colisions
        self._collisions = [] 
        for point in contactPoints:
            # pyb may have contacts with separation dist greater zero    
            if point[8] <= 0: 
                self._collisions.append((point[1], point[2]))

        # report collisions
        finalCollisions = [tup for tup in self._collisions if not any(val == 0 for val in tup)]
        if len(finalCollisions) > 0: 
            # print("Collisions:", finalCollisions)  # 0:plane, 1,5:robots
            pass

    def close(self) -> None:
        pyb.disconnect()
        


    ################################
    ### Create spawnable objects ###
    ################################
    def _spawn_robot(self, robot: Robot, env_idx: int) -> str:
        position = (robot.position + self._env_offsets[env_idx]).tolist()
        rotation = robot.orientation.tolist()
        urdf_path = self.asset_path + str(robot.urdf_path)
        name = robot.name if robot.name else "robot_" + str(len(self._robots))

        robotId = pyb.loadURDF(urdf_path, basePosition=position, baseOrientation=rotation, useFixedBase=True)
        
        # Track robot joints
        joints = []
        observableJoints = []
        controllableJoints = []
        jointNames = []
        initPos = [] 

        for i in range(pyb.getNumJoints(robotId)):
            info = pyb.getJointInfo(robotId, i) 

            jointName = info[12].decode('UTF-8')                # string name of the joint
            controllable = info[2]                              # moveable joints for executing a action
            jointAngle = pyb.getJointState(robotId, i)[0]       # initial angle for reset

            joints.append(i)
            jointNames.append(name + "/" + jointName)
            initPos.append(jointAngle)
            
            if controllable != 4:
                controllableJoints.append(i)                    # no fixed joint
            
            if jointName in robot.observable_joints:
                observableJoints.append((i, jointName))         # for speed up 

        # Track spawned robot
        self._robots.append((name, robotId, jointNames, joints, observableJoints, controllableJoints,initPos))       
        
        if robot.observable:
            self._observable_robots.append((name, robotId, jointNames, joints, observableJoints, controllableJoints, initPos))   

        return name


    def _spawn_cube(self, cube: Cube, env_idx: int) -> str:
        position = (cube.position + self._env_offsets[env_idx]).tolist()
        rotation = cube.orientation.tolist()
        name = cube.name if cube.name else "cube_" + str(len(self._obstacles))

        cube_id = pyb.createMultiBody(
            baseMass=.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in cube.scale], rgbaColor=cube.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in cube.scale]) if cube.collision else -1,
            basePosition=position,
            baseOrientation=rotation)
        
        # track spawned cube
        self._obstacles.append((name, cube_id, position, rotation))  
        
        if cube.observable:
            self._observable_obstacles.append((name, cube_id, position, rotation))   
        return name


    def _spawn_sphere(self, sphere: Sphere, env_idx: int) -> str:
        position = (sphere.position + self._env_offsets[env_idx]).tolist()
        name = sphere.name if sphere.name else "sphere_id_" + str(len(self._obstacles))
        rotation = [0.0, 0.0, 0.0, 1.0]

        sphere_id = pyb.createMultiBody(
            baseMass=.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=sphere.radius, rgbaColor=sphere.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=sphere.radius) if sphere.collision else -1,
            basePosition=position)
        
        # track spawned sphere
        self._obstacles.append((name, sphere_id, position, rotation)) 

        if sphere.observable:
             self._observable_obstacles.append((name, sphere_id, position, rotation)) 
        return name
    

    def _spawn_cylinder(self, cylinder: Cylinder, env_idx:int) -> str:
        position = (cylinder.position + self._env_offsets[env_idx]).tolist()
        rotation = [0.0, 0.0, 0.0, 1.0]
        name = cylinder.name if cylinder.name else "cylinder_id_" + str(len(self._obstacles))
        
        cylinder_id = pyb.createMultiBody(
            baseMass=.0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=cylinder.radius, length=cylinder.height, rgbaColor=cylinder.color),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=cylinder.radius, height=cylinder.height) if cylinder.collision else -1,
            basePosition=position, baseOrientation=rotation)

        # track spawned cylinder
        self._obstacles.append((name, cylinder_id, position, rotation))  

        if cylinder.observable:
            self._observable_obstacles.append((name, cylinder_id, position, rotation))  
        return name