
from scripts.envs.modular_env import ModularEnv
from scripts.envs.env_params import EnvParams

from scripts.spawnables.obstacle import Obstacle, Cube, Sphere, Cylinder
from scripts.spawnables.robot import Robot
from scripts.spawnables.urdf import Urdf

from scripts.envs.pybullet.robot import PyRobot
from scripts.envs.pybullet.obstacle import *

from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance, calc_distance
from scripts.rewards.timesteps import ElapsedTimesteps
from scripts.rewards.collision import Collision
from scripts.rewards.shaking import Shaking

from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.resets.collision_reset import CollisionReset
from scripts.resets.boundary_reset import BoundaryReset 

from stable_baselines3.common.vec_env.base_vec_env import *
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import math
import pybullet as pyb

import pandas as pd
import timeit

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
        self._collisionsCount: List[int] = np.zeros(params.num_envs)
        self.step_count = params.step_count
        self.verbose = params.verbose
        self.stepSize = params.step_size
        self.displayDelay = self.stepSize * self.robot_count

        # for the reset
        self._initRobots = params.robots
        self._initObstacles = params.obstacles
        self._initUrdfs = params.urdfs

        # save the distances in the current environment
        self._distance_funcions = []
        self._distances: Dict[str, np.ndarray] = {}
        self._distances_after_reset: Dict[str, np.ndarray] = {}
        self._last_dist: Dict[int, list] = {}
        self._accessDist = {}

        # save collidable objects for collision detection
        self._collidable = []
        self._obs = {}

        # calculate env offsets
        break_index = math.ceil(math.sqrt(self.num_envs))
        self._env_offsets = dict(zip(
            [i for i in range(params.num_envs)],
            [np.array([(i % break_index) * params.env_offset[0], math.floor(i / break_index) * params.env_offset[1], 0]) for i in range(params.num_envs)]
        ))


        # parameters to save execution times
        self.setupTime = 0
        self.actionTime = 0
        self.obsTime = 0

        self.log_dict = pd.DataFrame()

        # setup PyBullet simulation environment and interfaces
        self._setup_simulation(params.headless, params.step_size)

        ### allow tracking spawned objects ###
        self._robots: Dict[int, List[PyRobot]] = {}
        self._observable_robots: Dict[int, List[PyRobot]] = {}
        self._obstacles: Dict[int, List[PyObstacle]] = {}
        self._observable_obstacles: Dict[int, List[PyObstacle]] = {}
        self._observable_urdfs: Dict[int, List[Urdf]] = {}

        # setup rl environment
        self._setup_environments(params.robots, params.obstacles, params.urdfs)
        self._setup_rewards(params.rewards)
        self._setup_resets(params.rewards, params.resets)

        # init bace class last, allowing it to automatically determine action and observation space
        super().__init__(params) 

        # set dummy value to avoid KeyError
        if self.verbose > 0:
            self.set_attr("avg_rewards", 0)
            self.set_attr("avg_success", 0)
            self.set_attr("avg_resets", 0)

        if self.verbose > 1:
            self.set_attr("avg_setupTime", 0)
            self.set_attr("avg_actionTime", 0)    
            self.set_attr("avg_obsTime", 0) 

        if self.verbose > 2:
            for name, _ in self._distances_after_reset.items():
                self.set_attr("avg_" + name + "_euclid_dist" , 0)   
                self.set_attr("avg_" + name + "_anglular_dist" , 0)   
            self.set_attr("avg_steps", 0)
            self.set_attr("avg_coll", 0)


    def _setup_simulation(self, headless: bool, step_size: float):
        disp = pyb.DIRECT if headless else pyb.GUI
        pyb.connect(disp)
        pyb.setTimeStep(step_size)
        pyb.setGravity(0, 0, -9.8) 
        #pyb.setPhysicsEngineParameter(numSubSteps=1)
        #pyb.setRealTimeSimulation(0)
       

    def _setup_environments(self, robots: List[Robot], obstacles: List[Obstacle], urdfs: List[Urdf]) -> None:                                           
        # initialize dictionaries
        self._robots = {i: [] for i in range(self.num_envs)}
        self._observable_robots  = {i: [] for i in range(self.num_envs)}
        self._obstacles = {i: [] for i in range(self.num_envs)}
        self._observable_obstacles = {i: [] for i in range(self.num_envs)}
        self._observable_urdfs = {i: [] for i in range(self.num_envs)}

        # load ground plane
        pyb.loadURDF(self.asset_path  + "workspace/plane.urdf", [0,0,-1e-3])         

        # spawn objects for each environment
        for env_idx in range(self.num_envs):  
            # spawn urdfs
            for urdf in urdfs:
                self._spawn_urdf(urdf, env_idx)      
            # spawn robots
            for robot in robots:
                self._spawn_robot(robot, env_idx)
            # spawn obstacles
            for obstacle in obstacles:
                self._spawn_obstacle(obstacle, env_idx)  
 

    def _setup_rewards(self, rewards: List[Reward]) -> None:
        self._reward_fns = []

        for reward in rewards:
            if isinstance(reward, Distance):
                self._reward_fns.append(self._parse_distance_reward(reward))
            elif isinstance(reward, ElapsedTimesteps):
                self._reward_fns.append(self._parse_timestep_reward(reward))
            elif isinstance(reward, Collision):
                self._reward_fns.append(self._parse_collision_reward(reward))
            elif isinstance(reward, Shaking):
                self._reward_fns.append(self._parse_shaking_reward(reward))
            else:
                raise f"Reward {type(reward)} not implemented!"


    def _parse_distance_reward(self, distance: Distance):
        # get start and  end indicies for positions (x,y,z)
        obj1Pos = self._find_observable_object(distance.obj1,3)
        obj2Pos = self._find_observable_object(distance.obj2,3)
        
        # get start and  end indicies for angles (a,b,c,d)
        obj1Rot = self._find_observable_object(distance.obj1,4)
        obj2Rot = self._find_observable_object(distance.obj2,4)

        # extract name to allow created function to access it easily 
        name = distance.name    
        distance_weight = distance.distance_weight
        orientation_weight = distance.orientation_weight
        normalize = distance.normalize
        exponent = distance.exponent

        # parse function calculating distance
        def distance_per_env() -> Tuple[str, np.ndarray]:
            result = []

            for i in range(self.num_envs):                
                # extract x,y,z coordinates of objects
                pos1 = self._obs["Positions"][i][obj1Pos[0]:obj1Pos[1]]
                pos2 = self._obs["Positions"][i][obj2Pos[0]:obj2Pos[1]]
                
                # extract quaternion orientation from objects
                rot1 = self._obs["Rotations"][i][obj1Rot[0]:obj1Rot[1]]
                rot2 = self._obs["Rotations"][i][obj2Rot[0]:obj2Rot[1]]

                result.append(calc_distance(pos1, pos2, rot1, rot2))
            
            result = np.array(result)
            return name, result
    
        # add to existing distance functions
        self._distance_funcions.append(distance_per_env)

        def calculate_distance_reward():
            # get current distances
            distance_space, distance_orientation = self._get_distance_and_rotation(name)

            # apply weight factor
            weighted_space = distance_weight * (distance_space ** exponent)
            weighted_orientation = orientation_weight * (distance_orientation ** exponent)

            # skip normalization
            if not normalize:
                return weighted_space + weighted_orientation

            # retrieve distences after last reset
            begin_space, begin_orient = self._distances_after_reset[name]

            # calculate variance to previous distance, avoiding division by zero
            normalized_space = np.where(begin_space == 0, weighted_space, distance_weight * (distance_space ** exponent) / begin_space) 
            normalized_orient = np.where(begin_orient == 0, weighted_orientation, distance_orientation * (distance_orientation ** exponent) / begin_orient) 

            return normalized_space + normalized_orient

        return calculate_distance_reward
    
    def _parse_shaking_reward(self, shaking: Shaking):
        weight = shaking.weight
        length = shaking.length
        dist_name = shaking.distance_name
        
        for i in range(self.num_envs):
            self._last_dist[i] = []

        # reward elapsed timesteps
        def shaking_reward():
            # init enmtpy shaking array 
            shakings = np.zeros(self.num_envs)

            # get current distance
            distance_space, _ = self._get_distance_and_rotation(dist_name)

            # only save last length distances
            for i in range(self.num_envs):
                if len(self._last_dist[i]) > length:
                    self._last_dist[i].pop(0)

                # add recent distance if not already in there
                if not np.any(distance_space[i] == self._last_dist[i]):
                    self._last_dist[i].append(distance_space[i])

            # add negative reward for shaking
            for idx in range(self.num_envs):
                shaking = 0
                if len(self._last_dist[idx]) >= length:
                    flipping = []
                    for i in range(length-1):
                        flipping.append(0) if self._last_dist[idx][i+1] - self._last_dist[idx][i] >= 0 else flipping.append(1)
                    for j in range(length-2):
                        if flipping[j] != flipping[j+1]:
                            shaking += 1
                shakings[idx] = shaking

            return weight * shakings
        
        return shaking_reward
    
    def _parse_timestep_reward(self, elapsed: ElapsedTimesteps):
        """
        Reward elapsed timesteps according to the weight factor
        """
        weight = elapsed.weight
        
        def timestep_reward():
            return self._timesteps * weight
        
        return timestep_reward
    
    
    def _parse_collision_reward(self, collisionObj: Collision):
        """
        Punish collisions from collisionObj with any other collidable object according to the weight factor
        """
        objName = collisionObj.obj
        weight = collisionObj.weight
        
        def calculate_collision_reward() -> float:
            # get all collision
            result = []
            for envId in range(self.num_envs):
                coll = 0
                for robot in self._robots[envId]:
                    # only check collision for specified robot
                    if objName == robot.name:
                        if any(robot.id in tup for tup in self._collisions):
                            coll = 1
                result.append(coll) 

            result = np.array(result) * weight
            return result
        
        return calculate_collision_reward
    
    
    def _find_observable_object(self, name: str, obsSize: int) -> int:
        """
        Given the name of an observable object, tries to retrieve its index in the observations list. Since all 
        environments contain the same observable objects, we only iterate over the 1st environment. 

        Example: When two robots are observable and the position of the second robots is being queried, returns 
        start index and end index depending of the objectsize e.g. for (x,y,z) the size is 3
        """
        index = 0 

        # iterate over all robots and their joints 
        for robot in self._observable_robots[0]:
            if robot.name.endswith(name):
                return index * obsSize, (index * obsSize) + obsSize
            index += 1
            
            for jointName in robot.observableJointNames:
                if name.endswith(jointName):
                    return index * obsSize, (index * obsSize) + obsSize
                index += 1
                
        # iterate over all obstacles
        for obstacle in self._observable_obstacles[0]:
            if obstacle.name.endswith(name):
                return index * obsSize, (index * obsSize) + obsSize
            index += 1
        
        # if objext not found raise an error
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
            elif isinstance(reset, CollisionReset):
                self._reset_fns.append(self._parse_collision_reset(reset))
            elif isinstance(reset, BoundaryReset):
                self._reset_fns.append(self._parse_boundary_reset(reset))
            else:
                raise f"Reset {type(reset)} not implemented!"


    def _parse_distance_reset(self, reset: DistanceReset):
        # extract name to allot created function to access it easily
        name = reset.distance_name
        reward = reset.reward
        min_distance = reset.min_distance
        max_distance = reset.max_distance
        min_angle = reset.min_angle
        max_angle = reset.max_angle
        resetName = reset.name
        
        # parse function
        def reset_condition() -> np.ndarray:
            # get distances of current timestep
            distance, rotation = self._get_distance_and_rotation(name)  

            # add positive reward if condition successfully reached 
            if min_distance:
                dist_success = np.where(distance <= min_distance, True, False)  
        
                if min_angle:
                    rot_success = np.where(rotation <= min_angle, True, False)  
                    successes = np.logical_and(dist_success, rot_success)
                else:
                    successes = dist_success
                    
                resets = successes

                # apply reward in case of successes
                self._rewards += successes * reward 
            
            # apply penalty for reset condtion reached
            else:
                dist_resets = np.where(distance > max_distance, True, False)

                if max_angle:
                    rot_reset = np.where(rotation > max_angle, True, False)   
                    resets = np.logical_or(dist_resets, rot_reset)           
                else: 
                    resets = dist_resets
                
                successes = np.logical_not(resets)
                    
                # apply punishment/ reward in case of reset 
                self._rewards += resets * reward

            return resets, successes, resetName

        return reset_condition
    
    def _parse_boundary_reset(self, reset: BoundaryReset):
        min_bound = reset.min_bound
        max_bound = reset.max_bound
        objName = reset.obj
        reward = reset.reward
        name = reset.name

        # parse function
        def reset_condition() -> np.ndarray:   
            reset = []
            for envIdx in range(self.num_envs):
                for robot in self._observable_robots[envIdx]:                    
                    if robot.name == objName:
                        pos, _, _= robot.getPose()
                        jointsPos, _ = robot.getObservableJointsPose() 
                        pos = np.array(pos)
                        jointsPos = np.array(jointsPos)

                        resRobot = np.logical_or(pos < min_bound, pos > max_bound)
                        resJoints = np.logical_or(jointsPos < min_bound, jointsPos > max_bound)
                        reset.append(np.any(np.logical_or(resRobot, resJoints)))
            
            reset = np.array(reset)
            success = [not val for val in reset]
            self._rewards += reset * reward
            
            return reset, success, name
        
        return reset_condition


    def _parse_timesteps_reset(self, reset: TimestepsReset):
        max_steps = reset.max
        min_steps =  reset.min
        reward = reset.reward
        name = reset.name

        # parse function
        def reset_condition() -> np.ndarray:
            # signal reset whenever the current timespets exceed the max value
            resets = np.where(self._timesteps < max_steps, False, True)

            if min_steps:
                successes = np.where(self._timesteps >= min_steps, True, False) 
                self._rewards += successes * reward             
            else:
                successes = np.where(self._timesteps < max_steps, True, False)
                self._rewards += resets * reward

            return resets, successes, name
        
        return reset_condition

    def _parse_collision_reset(self, reset: CollisionReset):
        max_value = reset.max
        objName = reset.obj
        reward = reset.reward
        name = reset.name
        
        # parse function
        def reset_condition() -> np.ndarray:   
            for envId in range(self.num_envs):
                for robot in self._robots[envId]:
                    # only check collision for specified robot
                    if objName == robot.name:
                        if any(robot.id in tup for tup in self._collisions):
                            self._collisionsCount[envId] += 1

            # return true whenever more than max_value collisions occured  
            resets = np.where(self._collisionsCount > max_value, True, False)
            successes = np.where(self._collisionsCount < max_value, True, False)
            
            self._rewards += resets * reward    # punish collision
            return resets, successes, name
        
        return reset_condition

    def _move_robot_via_velocity(self, robot: PyRobot, action: np.ndarray) -> None: 
        ''' 
        control via joint velocities
        if we use physics sim, the engine can deal with those on its own
        if we don't, we run simple algebra to get the new joint angles for this step and then apply them
        '''
        newVel = action * robot.maxVelocity                 # transform action (-1 to 1) to desired new joint angles
        
        if self.headless:
            jointDelta = newVel * self.stepSize             # compute delta for this sim step size
            newJoint = jointDelta + robot.getJointAngles()  # add delta to current joint angles
            pybAngle = [[value] for value in newJoint]      # transform format for pybullet

            # execute movement
            pyb.resetJointStatesMultiDof(bodyUniqueId=robot.id, jointIndices=robot.controllableJoints, 
                                         targetValues=pybAngle)
        else: 
            # use engine to apply velocities to robot
            pyb.setJointMotorControlArray(bodyUniqueId=robot.id, jointIndices=robot.controllableJoints,
                                          controlMode=pyb.VELOCITY_CONTROL, targetVelocities=newVel)


    def _move_robot_via_position(self, robot: PyRobot, action: np.ndarray) -> None: 
        ''' 
        control via joint angles
        actions are the new desired joint angles themselves
        '''
        # transform action (-1 to 1) to desired new joint angles
        joints_range = robot.upper - robot.lower
        desiredAngles = action * (joints_range / 2) + (robot.lower + robot.upper) / 2

        # if we don't use physics sim, which will only perform step towards desired new joints, we have to
        # clamp new joint angles such that they move with at most the maximum velocity within the next sim step
        if self.headless:
            # compute maximum step we do in that direction
            jointDelta = desiredAngles - robot.getJointAngles()
            jointDist = np.linalg.norm(jointDelta)
            jointDist = jointDist if jointDist != 0 else 1
            jointDelta = jointDelta / jointDist
            step_times_velocity = np.min(robot.maxVelocity) * self.stepSize
            if jointDist > step_times_velocity:
                joint_mul = step_times_velocity
            else:
                joint_mul = jointDist
            jointDelta = jointDelta * joint_mul
           
            desiredAngles = jointDelta + robot.getJointAngles()  # compute joint angles we can actually go to
            pybAngle = [[value] for value in desiredAngles]      # transform format for pybullet

            # execute movement
            pyb.resetJointStatesMultiDof(bodyUniqueId=robot.id, jointIndices=robot.controllableJoints, targetValues=pybAngle)

        else: 
            # set joint tragets for simulation
            pyb.setJointMotorControlArray(bodyUniqueId=robot.id, jointIndices=robot.controllableJoints, 
                                          controlMode=pyb.POSITION_CONTROL, targetPositions=desiredAngles)
    

    def step_async(self, actions: np.ndarray) -> None:
        """
        This function perfomrs certain actions for each roboter in an environments. It iterates over each environment 
        and moves the joints of all robots according to the action type and values. Then it performs a collion detection
        and shwos all collision in the console.
        """
        startTime = timeit.default_timer()

        # get all robots from an environment and perform actions
        for envId in range(self.num_envs):
            for i, robot in enumerate(self._robots[envId]):
                
                if robot.control_type == "Velocity":
                    self._move_robot_via_velocity(robot, actions[envId][i*len(robot.limits):len(robot.limits)*(i+1)]) 
                
                elif robot.control_type == "Position":
                    self._move_robot_via_position(robot, actions[envId][i*len(robot.limits):len(robot.limits)*(i+1)])           
                
                else:
                    raise Exception(f"Control type {robot.control_type} not implemented!")

            # update obstacles that have a trajectory    
            for obstacle in self._obstacles[envId]:
                obstacle.update()
            
        self.actionTime = timeit.default_timer()  - startTime    
            	
        # step simulation amount of times according to params
        for _ in range(self.step_count):
            pyb.stepSimulation()  
            #time.sleep(self.displayDelay)
                
        # update the collision model if necessary
        if self.headless:
            pyb.performCollisionDetection()           

    
    def step_wait(self) -> VecEnvStepReturn:
        self._timesteps += 1                        # increment elapsed timesteps
        self._obs = self._get_observations()        # get observations and track time     
        self._distances = self._get_distances()     # get distances after updated observations
        self._collisions = self._get_collisions()   # get collisions
        self._rewards = self._get_rewards()         # get rewards after updated distances and collisions

        # check if an environment is dones (needs a reset) or successfully finished (needs reset)
        resets = np.full(self.num_envs, False)
        successes = np.full(self.num_envs, True)       
       
        # go over each reset condition
        donesDict = {}
        logNames = []
        logResets = []
        logSuccesses = []
        for fn in self._reset_fns:
            curr_reset, curr_success, name = fn()
            resets = np.logical_or(resets, curr_reset)
            successes = np.logical_and(successes, curr_success)
            logNames.append(name)
            logResets.append(curr_reset)
            logSuccesses.append(curr_success)

        if self.verbose > 3: 
            for currName, currReset, currSuccess in zip(logNames, logResets, logSuccesses):
                for idx in range(self.num_envs):

                    if idx not in donesDict:
                        donesDict[idx] = {}
                    
                    donesDict[idx][f"{currName}_reset"] = currReset[idx]
                    donesDict[idx][f"{currName}_success"] = currSuccess[idx]
                                      
        # get environemnt idx that need a reset 
        #self._dones = np.logical_or(resets, successes)
        self._dones = resets
        reset_idx = np.where(self._dones)[0]             

        # create csv file with informations about each specific environment each timestep
        if self.verbose > 3:
            for envIdx in range(self.num_envs):
                # extract end effector pos
                positions = []
                for robot in self._observable_robots[envIdx]:                    
                        jointsPos, _ = robot.getObservableJointsPose() 
                        positions.append(list(jointsPos))
                info = {
                    "env_id": envIdx,
                    "timestep": self._timesteps[envIdx], 
                    "joints_pos": positions,
                    "reward": self._rewards[envIdx],
                    "collision": self._collisionsCount[envIdx],
                    "avg_setupTime": self.setupTime/self.num_envs,
                    "avg_actionTime": self.actionTime/self.num_envs,
                    "avg_obsTime": self.obsTime/self.num_envs,
                    **self._accessDist[envIdx],
                    **donesDict[envIdx]
                }
                self.log_dict = pd.concat([self.log_dict, pd.DataFrame([info])], ignore_index=True)

        # apply resets
        if reset_idx.size > 0: 
            # Log only general information averaged over all environments
            if self.verbose > 0:   
                self.set_attr("avg_rewards", np.average(self._rewards[reset_idx])) 
                
                success_rate = np.sum(successes) * (1/self.num_envs)
                resets_rate = 1 - success_rate
                self.set_attr("avg_success", success_rate)    
                self.set_attr("avg_resets",  resets_rate)   
            
            # Add info about execution times averaged over all environments
            if self.verbose > 1:
                self.set_attr("avg_setupTime", self.setupTime)
                self.set_attr("avg_actionTime", self.actionTime)    
                self.set_attr("avg_obsTime", self.obsTime)         

            # Add information about rewards and resets averaged over all environments
            if self.verbose > 2:
                for dist_name, distance in self._distances.items():
                    euclid_dist = np.array([value[0] for value in distance])
                    angular_dist = np.array([value[1] for value in distance])

                    self.set_attr("avg_" + dist_name + "_euclid_dist" , np.average(euclid_dist[reset_idx]))
                    self.set_attr("avg_" + dist_name + "_anglular_dist" , np.average(angular_dist[reset_idx])) 
                
                self.set_attr("avg_coll", np.sum(self._collisionsCount[reset_idx]) * (1/self.num_envs)) 
                self.set_attr("avg_steps", np.average(self._timesteps[reset_idx]))        
            
            self.reset(reset_idx)

        return self._obs, self._rewards, self._dones, self.env_data


    def reset(self, env_idxs: np.ndarray=None) -> VecEnvObs:
        startTime = timeit.default_timer()
        
        # reset entire simulation
        if env_idxs is None:
            pyb.resetSimulation()  
            self._setup_environments(self._initRobots, self._initObstacles, self._initUrdfs)    # build environment new       
            self._timesteps = np.zeros(self.num_envs)                                           # reset timestep tracking
            self._collisionsCount = np.zeros(self.num_envs)                                     # reset collisions count tracking
            self._obs = self._get_observations()                                                # reset observations 
            self._distances_after_reset = self._get_distances()                                 # calculate new distances
        
        # reset envs manually
        else:           
            # select each environment
            for i in env_idxs:
                # reset all robots and all obstacles 
                for robot in self._robots[i]:
                    robot.reset()
                for obstacle in self._obstacles[i]:
                    obstacle.reset()

            self._timesteps[env_idxs] = 0           # reset timestep tracking
            self._collisionsCount[env_idxs] = 0     # reset collisions count tracking

            # reset observations # todo: only recalculate necessary observations
            self._obs = self._get_observations()

            # note new distances # todo: only recalculate necessary distances
            self._distances_after_reset = self._get_distances()

        self.setupTime = timeit.default_timer() - startTime
        return self._obs

    def get_robot_dof_limits(self) -> List[Tuple[float, float]]:
        """
        Extract the joint boundaries from each robot in the first environment, since the other environments contain 
        exactly the same robots. 
        """
        limits = [] # init array
        for robot in self._robots[0]:
            if robot.control_type == "Position":
                limits += robot.limits
            elif robot.control_type == "Velocity":
                for vel in robot.maxVelocity:
                    limits.append((-vel,vel))
            else:
                raise Exception(f"Unknown control type: {robot.control_type}")
        
        return limits


    def _get_observations(self) -> VecEnvObs:
        startTime = timeit.default_timer()

        # create empty arrays, allowing obs to be appended
        positions = np.array([])
        rotations = np.array([])
        scales = np.array([])
        joint_positions = np.array([])

        # iterate through each environment
        for env_idx in range(self.num_envs):
            # get observations from all robots and their joints in the environment
            for robot in self._observable_robots[env_idx]:
                pos, rot, scale = robot.getPose()

                # add robot pos, rotation and scale to list of observations
                positions = np.append(positions, pos)
                rotations = np.append(rotations, rot)
                scales = np.append(scales, scale)

                # get observations from all observable joints from the robot
                jointsPos, jointsRot = robot.getObservableJointsPose()           

                # add pos and rotation to list of observations
                positions = np.append(positions, jointsPos)
                rotations = np.append(rotations, jointsRot)

                # get angles of joints
                angles = robot.getJointAngles()
                joint_positions = np.append(joint_positions, angles)

            # get observations from all obstacles in environment
            for obstacle in self._observable_obstacles[env_idx]:
                pos, rot, scale = obstacle.getPose() 
                
                positions = np.append(positions, pos)
                rotations = np.append(rotations, rot)
                scales = np.append(scales, scale)

            # get observations from all observable urdfs
            for urdf in self._observable_urdfs[env_idx]:
                pos, rot = pyb.getBasePositionAndOrientation(urdf[0]) 
                pos -= urdf[1]
                scale =  pyb.getVisualShapeData(urdf[0])[3][3]

                positions = np.append(positions, pos)
                rotations = np.append(rotations, rot)
                scales = np.append(scales, scale)

        # reshape observations
        positions = positions.reshape(self.num_envs, -1)
        rotations = rotations.reshape(self.num_envs, -1)
        scales = scales.reshape(self.num_envs, -1)
        joint_positions = joint_positions.reshape(self.num_envs, -1)

        endTime = timeit.default_timer()
        self.obsTime = endTime - startTime
        
        return {
            "Positions": positions,
            "Rotations": rotations,
            "Scales": scales,
            "JointPositions": joint_positions
        }
    

    def _get_distance_and_rotation(self, name: str) -> Tuple[float, float]:
        """
          return current distance_space (meters), distance_orientation (angle)
        """
        # get current distances
        distances = self._distances[name]

        # return distance_space (meters), distance_orientation (angle)
        return distances[:, 0], distances[:, 1]


    def _get_distances(self) -> Dict[str, np.ndarray]:
        distances = {}      # reset current distances
        
        for distance_fn in self._distance_funcions:
            name, distance = distance_fn()      # calcualte current distances
            distances[name] = distance
            
            # if needed rearrange format
            if self.verbose > 2:
                self._accessDist = {}
                for idx in range(self.num_envs):
                    self._accessDist[idx] = {
                        f"{name}_dist_euclid": distance[idx][0],
                        f"{name}_dist_angular": distance[idx][1],
                    }

        return distances


    def _get_rewards(self) -> List[float]:
        """
        Get the rewards for all environments of the simulation that are added to the rewards list
        """
        rewards = np.zeros(self.num_envs)
        for fn in self._reward_fns:
            rewards += fn()

        return rewards


    def _get_dones(self) -> List[bool]:
        dones = np.full(self.num_envs, False)    # init default array: No env is done
        successes = np.full(self.num_envs, False)

        # check if any of the functions specify a reset or success
        for fn in self._reset_fns:
            curr_dones, curr_success = fn()
            dones = np.logical_or(dones, curr_dones)
            successes = np.logical_and(successes, curr_success)

        # reset environments where dones == True or a success occured   
        reset_needed = np.logical_or(dones, successes)
        reset_idx = np.where(reset_needed)[0]

        # log and reset environments if necessary
        if reset_idx.size > 0:
            if self.verbose > 0:
                # log success of environments
                self.set_attr("average_success", np.average(successes[reset_idx]))

            if self.verbose > 1:
                # log distances of environments
                for name, distance in self._distances.items():
                    # dont take rotation into account
                    curr_dist = [value[0] for value in distance[reset_idx]]
                    self.set_attr("average/" + name + "_dist" , np.average(curr_dist))
            
                # log average steps of environments 
                self.set_attr("average_steps", np.average(self._timesteps[reset_idx]))

                # log average collisions of environments  
                self.set_attr("average_collision", np.average(self._collisionsCount[reset_idx]))
            
            self.reset(reset_idx)

        return dones
    
    def _get_collisions(self) -> List[bool]:
        """
        Returns a List containing all collision of all environments 
        """   
        _contactPoints = pyb.getContactPoints()  # get collisions

        if not _contactPoints: return [] # no contacts made

        # extract all colisions
        _collisions = [] 
        for point in _contactPoints:
            # pyb may have contacts with separation dist greater zero    
            if point[8] <= 0: 
                if point[1] in self._collidable and point[2] in self._collidable:
                    _collisions.append((point[1], point[2]))

        # remove duplicated collision
        _collisions = list(set(_collisions))  

        return _collisions

    def _on_contact_report_event(self) -> None:
        """
        Reports all contacts a robot has while moving.
        """   
        contactPoints = pyb.getContactPoints()  # get collisions
        if not contactPoints: return # skip if there are no collisions

        # extract all colisions
        self._collisions = [] 
        for point in contactPoints:
            # pyb may have contacts with separation dist greater zero    
            if point[8] <= 0: 
                self._collisions.append((point[1], point[2]))

        # report collisions
        finalCollisions = [tup for tup in self._collisions if not any(val == 0 for val in tup)]        
        if finalCollisions: 
            print("Collisions:", finalCollisions)   # 0:plane, 1:table, 5:robots
            for coll in finalCollisions:
                print(f'Collision from {pyb.getBodyInfo(coll[0])[1]} with {pyb.getBodyInfo(coll[1])[1]}')


    def close(self, path:str=None) -> None:
        """ 
        Shut down the simulation 
        """
        pyb.disconnect()                        # close sim
        if self.verbose > 3:
            if not path:
                return 0
            
            df = pd.DataFrame(self.log_dict)        # transform logs to a df       
            df.to_csv(path +".csv", index=False)    # save df to a csv file
                

    def _spawn_robot(self, robot: Robot, env_idx: int) -> str:
        """ 
        Spawn a robot object into the environment and safe it in a dictionary with its environment id 
        """
        urdf_path = self.asset_path + str(robot.urdf_path)
        newRobot = PyRobot(urdf_path, robot.control_type, robot.max_velocity, robot.observable_joints, robot.name, self._env_offsets[env_idx], robot.position, 
                           robot.orientation[::-1], robot.collision, robot.observable)

        # track spawned robot
        self._robots[env_idx].append(newRobot)          
        
        if robot.observable:
            self._observable_robots[env_idx].append(newRobot)    
        
        if robot.collision:
            self._collidable.append(newRobot.id)   

        return newRobot.name
    
    def _spawn_urdf(self, urdf: Urdf, env_idx: int) -> str:
        """ 
        Spawn an urdf object into the environment and safe it in a dictionary with its environment id 
        """
        urdf_path = self.asset_path + str(urdf.urdf_path)
        urdf_pos = (urdf.position + self._env_offsets[env_idx]).tolist()
        urdf_ori = urdf.orientation[::-1]

        # create pybullet instance of urdf
        new_urdf = pyb.loadURDF(urdf_path, basePosition=urdf_pos, baseOrientation=urdf_ori, 
                               useFixedBase=True, globalScaling=urdf.scale[0])
        
        if urdf.observable:
            self._observable_urdfs[env_idx].append((new_urdf, self._env_offsets[env_idx]))

        if urdf.collision:
            self._collidable.append(new_urdf)

        return urdf.name


    def _spawn_obstacle(self, obstacle: Obstacle, env_idx: int) -> str:
        """ 
        Spawn a obstacle object into the environment and safe it in a dictionary with its environment id 
        """
        newObject = None
        orientation = obstacle.orientation[::-1]

        if isinstance(obstacle, Cube):
            newObject = PyCube(obstacle.name, self._env_offsets[env_idx], obstacle.position, orientation, 
                               obstacle.scale, obstacle.static, obstacle.collision, obstacle.color, self.step_count, 
                               self.stepSize, obstacle.endpoint, obstacle.velocity)
        elif isinstance(obstacle, Sphere):
            newObject = PySphere(obstacle.radius, obstacle.name, self._env_offsets[env_idx], obstacle.position, 
                                 orientation, obstacle.static, obstacle.collision, obstacle.color, self.step_count, 
                                 self.stepSize, obstacle.endpoint, obstacle.velocity)
        elif isinstance(obstacle, Cylinder):
            newObject = PyCylinder(obstacle.radius, obstacle.height, obstacle.name, self._env_offsets[env_idx], 
                                   obstacle.position, orientation,obstacle.static, obstacle.collision, obstacle.color, 
                                   self.step_count, self.stepSize, obstacle.endpoint, obstacle.velocity)
        else:
            raise f"Obstacle {type(obstacle)} not implemented"
        
        # track spawned object
        self._obstacles[env_idx].append(newObject)
        
        if obstacle.observable:
            self._observable_obstacles[env_idx].append(newObject) 

        if obstacle.collision:
            self._collidable.append(newObject.id)

        return newObject.name