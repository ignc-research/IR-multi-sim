import math
from typing import List, Tuple, Union
from scripts.envs.modular_env import ModularEnv
from scripts.envs.params.env_params import EnvParams
from scripts.rewards.distance import Distance, calc_distance
from scripts.rewards.timesteps import ElapsedTimesteps
from scripts.spawnables.obstacle import Obstacle, Cube, Sphere, Cylinder
from scripts.spawnables.robot import Robot
from scripts.rewards.reward import Reward
from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
from scripts.envs.params.control_type import ControlType
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import *
from pathlib import Path


def _add_position_offset(pos: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], offset: np.ndarray):
    if isinstance(pos, Tuple):
        return pos[0] + offset, pos[1] + offset
    return pos + offset


class IsaacEnv(ModularEnv):
    def __init__(self, params: EnvParams) -> None:
        """
        asset_path: relative path to root asset folder.
        step_size: amount of steps simulated before RL model is queried for new actions.
        headless: False if the simulation is not supposed to be rendered, otherwise True.
        robots: List of robots to spawn in each environment.
        obstacles: List of obstacles to spawn in each environment.
        rewards: List of rewards used to calculate total reward for each environment
        num_envs: Number of concurrently simulated environments.
        offset: Space between environments to prevent interference.
        """
        # setup asset path to allow importing robots
        self.asset_path = Path().absolute().joinpath(params.asset_path)

        # setup basic information about simulation
        self.num_envs = params.num_envs
        self.robot_count = len(params.robots)
        self.observable_robots_count = len([r for r in params.robots if r.observable])
        self.observable_robot_joint_count = sum(len(r.observable_joints) for r in params.robots)
        self.obstacle_count = len(params.obstacles)
        self.observable_obstacles_count = len([o for o in params.obstacles if o.observable])
        self._timesteps: List[int] = np.zeros(params.num_envs)
        self.step_count = params.step_count
        self.control_type = params.control_type
        self.verbose = params.verbose

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

        # setup ISAAC simulation environment and interfaces
        self._setup_simulation(params.headless, params.step_size)
        self._setup_urdf_import()
        self._setup_physics()

        # allow tracking spawned objects
        from omni.isaac.core.articulations import Articulation
        self._robots: List[Articulation] = []
        self._observable_robots: List[Articulation] = []
        self._observable_robot_joints: List[Articulation] = []
        
        from omni.isaac.core.prims.geometry_prim import GeometryPrim
        self._obstacles: List[Tuple[GeometryPrim, Obstacle]] = []
        # contains list of observable obstacles and observable robot joints
        self._observable_obstacles: List[GeometryPrim] = []

        # setup rl environment
        self._setup_environments(params.robots, params.obstacles)
        self._setup_rewards(params.rewards)
        self._setup_resets(params.rewards, params.resets)

        # init bace class last, allowing it to automatically determine action and observation space
        super().__init__(params)
    
    def _setup_simulation(self, headless: bool, step_size: float):
        # isaac imports may only be used after SimulationApp is started (ISAAC uses runtime plugin system)
        from omni.isaac.kit import SimulationApp
        self._simulation = SimulationApp({"headless": headless})

        # make sure simulation was started
        assert self._simulation is not None, "Isaac Sim failed to start!"
        assert self._simulation.is_running(), "Isaac Sim failed to start!"

        # terminate simulation once program exits
        import atexit
        atexit.register(self._simulation.close)

        # retrieve interfaces allowing to access Isaac
        from omni.usd._usd import UsdContext
        self._context: UsdContext = self._simulation.context
        self._app = self._simulation.app

        # create a world, allowing to spawn objects
        from omni.isaac.core import World
        self._world = World(physics_dt=step_size) # todo: find default parameter in env_args for physics_dt=step_size
        self._scene = self._world.scene
        self._stage = self._world.stage
        
        assert self._world != None, "Isaac world failed to load!"
        assert self._scene != None, "Isaac scene failed to load!"
        assert self._stage != None, "Isaac stage failed to load!"

    def _setup_urdf_import(self):
        # configure urdf importer
        from omni.kit.commands import execute
        result, self._config = execute("URDFCreateImportConfig")
        if not result:
            raise "Failed to create URDF import config"
        
        # set defaults in import config
        self._config.merge_fixed_joints = False
        #self._config.convex_decomp = False
        #self._config.import_inertia_tensor = True
        self._config.fix_base = True
        self._config.make_default_prim = True
        #self._config.create_physics_scene = True

    def _setup_physics(self):
        # setup physics
        # subscribe to physics contact report event, this callback issued after each simulation step
        from omni.physx import get_physx_simulation_interface
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

        # track collisions
        self._collisions: List[Tuple(int, int)] = []

        # configure physics simulation
        from omni.physx.scripts.physicsUtils import UsdPhysics, UsdShade, Gf
        scene = UsdPhysics.Scene.Define(self._stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        # Configure default floor material
        self._floor_material_path = "/floorMaterial"
        UsdShade.Material.Define(self._stage, self._floor_material_path)
        floor_material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._floor_material_path))
        floor_material.CreateStaticFrictionAttr().Set(0.8)
        floor_material.CreateDynamicFrictionAttr().Set(0.8)
        floor_material.CreateRestitutionAttr().Set(0.1)

        # Configure default collision material
        self._collision_material_path = "/collisionMaterial"
        UsdShade.Material.Define(self._stage, self._collision_material_path)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._collision_material_path))
        material.CreateStaticFrictionAttr().Set(0.5)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0.1)
        material.CreateDensityAttr().Set(0.001) 

        # setup ground plane
        ground_prim_path = "/World/defaultGroundPlane"
        self._scene.add_default_ground_plane(prim_path=ground_prim_path)

        # add collision to ground plane
        self._add_collision_material(ground_prim_path, self._floor_material_path)

    def _setup_environments(self, robots: List[Robot], obstacles: List[Obstacle]) -> None:
        # spawn objects for each environment
        for env_idx in range(self.num_envs):    
            # spawn robots
            for robot in robots:
                # import robot from urdf
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
            else:
                raise Exception(f"Reward {type(reward)} not implemented!")
        
    def _parse_distance_reward(self, distance: Distance):
        # parse indices in observations
        obj1_start, _ = self._parse_observable_object_range(distance.obj1)
        obj2_start, _ = self._parse_observable_object_range(distance.obj2)

        # extract name to allow created function to access it easily
        name = distance.name
        weight = distance.weight
        normalize = distance.normalize

        # create function calculating distance
        def distance_per_env() -> Tuple[str, np.ndarray]:
            # calculate distances as np array
            result = []
            for i in range(self.num_envs):
                result.append(calc_distance(
                    # extract x,y,z coordinates of objects
                    self._obs[i][obj1_start:obj1_start+3],
                    self._obs[i][obj2_start:obj2_start+3]
                ))
            result = np.array(result)

            return name, result

        # add to existing distance functions
        self._distance_funcions.append(distance_per_env)
    
        def calculate_distance_reward():
            # get current distances
            distance = self._distances[name]

            # apply weight factor
            weighted_distance = distance * weight

            # skip normalization
            if not normalize:
                return weighted_distance

            # retrieve distences after last reset
            beginning_distances = self._distances_after_reset[name]

            # calculate variance to previous distance, avoiding division by zero
            return np.where(beginning_distances == 0, weighted_distance, weight * distance / beginning_distances)

        return calculate_distance_reward
    
    def _parse_timestep_reward(self, elapsed: ElapsedTimesteps):
        weight = elapsed.weight

        # reward elapsed timesteps
        def timestep_reward():
            return self._timesteps * weight
        
        return timestep_reward

    def _parse_observable_object_range(self, name: str) -> Tuple[int, int]:
        """
        Given the name of an observable object, tries to retrieve its beginning and end position index in the observation buffer
        """
        index = self._find_observable_object(name)

        # calculate start index (multiply by 7: xyz for position, quaternion for rotation)
        start = index * 7

        # return start and end index
        return start, start + 6

    def _find_observable_object(self, name: str) -> int:
        """
        Given the name of an observable object, tries to retrieve its index in the observations list.
        Example: When two robots are observable and the position of the second robots is being queried, returns index 1.
        """
        # robots are input first into observations
        for index, robot in enumerate(self._observable_robots):
            if robot.name.endswith(name):
                return index
        
        # observable joints second
        for index, joint in enumerate(self._observable_robot_joints):
            if joint.name.endswith(name):
                return index + self.observable_robots_count

        # obstacles third
        for index, obstacle in enumerate(self._observable_obstacles):
            if obstacle.name.endswith(name):
                return index + self.observable_robots_count + self.observable_robot_joint_count

        raise Exception(f"Object {name} must be observable if used for reward")

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
        min_value = reset.min
        max_value = reset.max

        # parse function
        def reset_condition() -> np.ndarray:
            # get distances of current timestep
            d = self._distances[name]

            # return true whenever the distance exceed max or min value
            return np.where(min_value <= d, np.where(d <= max_value, False, True), True)

        return reset_condition

    def _parse_timesteps_reset(self, reset: TimestepsReset):
        max_value = reset.max

        # parse function
        def reset_condition() -> np.ndarray:
            # return true whenever the current timespets exceed the max value
            return np.where(self._timesteps < max_value, False, True)

        return reset_condition

    def step_async(self, actions: np.ndarray) -> None:
        # print("Actions:", actions)

        # apply actions
        if self.control_type == ControlType.Velocity:
            # set joint velocities
            for i, robot in enumerate(self._robots):
                robot.set_joint_velocities(actions[i])
        elif self.control_type == ControlType.Position:
            # set joint positions
            for i, robot in enumerate(self._robots):
                robot.set_joint_positions(actions[i])
        else:
            raise Exception(f"Control type {self.control_type} not implemented!")
            
        # step simulation amount of times according to params
        for _ in range(self.step_count):
            self._simulation.update()
    
    def step_wait(self) -> VecEnvStepReturn:
        # get observations
        self._obs = self._get_observations()

        # calculate current distances after observations were updated
        self._distances = self._get_distances()

        # calculate rewards after distances were updated
        self._rewards = self._get_rewards()

        # get dones
        self._dones = self._get_dones()

        # print("Obs    :", self._obs)
        # print("Dist.  :", self._distances["TargetDistance"])
        # print("Rewards:", self._rewards)
        # print("Dones  :", self._dones)
        # print("Timest.:", self._timesteps)

        # log rewards
        if self.verbose > 0:
            self.set_attr("average_rewards", np.average(self._rewards))

            # log distances
            if self.verbose > 1:
                for name, distance in self._distances.items():
                    self.set_attr("distance_"+name, np.average(distance))


        return self._obs, self._rewards, self._dones, self.env_data

    def reset(self, env_idxs: np.ndarray=None) -> VecEnvObs:
        # reset entire simulation
        if env_idxs is None:
            # initialize sim
            self._world.reset()

            # reset timesteps
            self._timesteps = np.zeros(self.num_envs)
        else:
            # select each environment
            for i in env_idxs:
                # reset all obstacles to default pose
                for geometryPrim, _ in self._get_obstacles(i):
                    # default case: reset obstacle to default position without randomization
                    geometryPrim.post_reset()

                # reset all robots to default pose
                for robot in self._get_robots(i):
                    robot.post_reset()

            # reset timestep tracking
            self._timesteps[env_idxs] = 0

        # reset observations # todo: only recalculate necessary observations
        self._obs = self._get_observations()

        # note new distances # todo: only recalculate necessary distances
        self._distances_after_reset = self._get_distances()
        
        return self._obs

    def get_robot_dof_limits(self) -> List[Tuple[float, float]]:
        # init array
        limits = []

        # ony get dof limits from robots of first environment
        for i in range(self.robot_count):
            for limit in self._robots[i].get_articulation_controller().get_joint_limits():
                limits.append(list(limit))
        
        return limits
    
    def _get_robots(self, env_idx: int):
        start_idx = env_idx * self.robot_count

        return [self._robots[i] for i in range(start_idx, start_idx + self.robot_count)]

    def _get_obstacles(self, env_idx: int):
        start_idx = env_idx * self.obstacle_count

        return [self._obstacles[i] for i in range(start_idx, start_idx + self.obstacle_count)]

    def close(self) -> None:
        self._simulation.close()

    def _get_observations(self) -> VecEnvObs:
        obs = []

        # iterate through each env
        for env_idx in range(self.num_envs):
            env_obs = []

            # get observations from all robots in environment
            robot_idx_offset = self.observable_robots_count * env_idx
            for robot_idx in range(robot_idx_offset, self.observable_robots_count + robot_idx_offset):
                # get robot of environment
                robot = self._observable_robots[robot_idx]

                # get its pose
                pos, rot = robot.get_local_pose()
                # apply env offset
                pos -= self._env_offsets[env_idx]

                # add robot pos and rotation to list of observations
                env_obs.extend(pos)
                env_obs.extend(rot)
                env_obs.extend(robot.get_local_scale())

            # get observations from all observable joints in environment
            joint_idx_offset = self.observable_robot_joint_count * env_idx
            for joint_idx in range(joint_idx_offset, self.observable_robot_joint_count + joint_idx_offset):
                # get joint of environment
                joint = self._observable_robot_joints[joint_idx]

                # get its pose
                pos, rot = joint.get_local_pose()

                # add pos and rotation to list of observations
                env_obs.extend(pos)
                env_obs.extend(rot)

            # get observations from all obstacles in environment
            obstacle_idx_offset = self.observable_obstacles_count * env_idx
            for obstacle_idx in range(obstacle_idx_offset, self.observable_obstacles_count + obstacle_idx_offset):
                # get obstacle of environment
                obstacle = self._observable_obstacles[obstacle_idx]

                # get its pose
                pos, rot = obstacle.get_local_pose()
                # apply env offset
                pos -= self._env_offsets[env_idx]

                # add obstacle pos and rotation to list of observations
                env_obs.extend(pos)
                env_obs.extend(rot)
                env_obs.extend(obstacle.get_local_scale())

            # add observations gathered in environment to dictionary
            obs.append(env_obs)

        return np.array(obs)

    def _get_distances(self) -> Dict[str, np.ndarray]:
        # reset current distances
        distances = {}

        for distance_fn in self._distance_funcions:
            # calcualte current distances
            name, distance = distance_fn()

            distances[name] = distance

        return distances

    def _get_rewards(self) -> List[float]:
        rewards = np.zeros(self.num_envs)

        for fn in self._reward_fns:
            rewards += fn()

        return rewards

    def _get_dones(self) -> List[bool]:
        # init default array: No environment is done
        dones = np.full(self.num_envs, False)

        # check if any of the functions specify a reset
        for fn in self._reset_fns:
            dones = np.logical_or(dones, fn())

        # increment elapsed timesteps if environment isn't done
        self._timesteps = np.where(dones, 0, self._timesteps + 1)

        # reset environments where dones == True
        reset_idx = np.where(dones)[0]

        # reset environments if necessary
        if reset_idx.size > 0:
            self.reset(reset_idx)

        return dones

    def _on_contact_report_event(self, contact_headers, contact_data):
        """
        After each simulation step, ISAAC calles this function. 
        Parameters contain updates about the collision status of spawned objects
        """
        # import required class
        from omni.physx.scripts.physicsUtils import PhysicsSchemaTools

        # clear collisions of previous step
        self._collisions = []

        for contact_header in contact_headers:
            # parse contact information
            contact_type = str(contact_header.type)

            # prim paths of objects with updated collision status
            actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

            # contact was found
            if 'CONTACT_FOUND' in contact_type or 'CONTACT_PERSIST' in contact_type:
                self._collisions.append((actor0, actor1))

        if(len(self._collisions) > 0):
            print(self._collisions)

    def _spawn_robot(self, robot: Robot, env_idx: int) -> str:
        """
        Loads in a URDF file into the world at position and orientation.
        """
        abs_path = str(self._get_absolute_asset_path(robot.urdf_path))

        # import URDF to temporary scene
        from omni.kit.commands import execute
        success, prim_path = execute(
            "URDFParseAndImportFile", 
            urdf_path=abs_path, 
            import_config=self._config
        )

        # make sure import succeeded
        assert success, "Failed urdf import of: " + abs_path

        # move robot to desired location
        from omni.isaac.core.articulations import Articulation
        obj = Articulation(prim_path, f"env{env_idx}-{robot.name}", robot.position + self._env_offsets[env_idx], orientation=robot.orientation)
        self._scene.add(obj)

        # track spawned robot
        self._robots.append(obj)

        # add it to list of observable objects, if necessary
        if robot.observable:
            self._observable_robots.append(obj)

        # configure collision
        if robot.collision:
            self._add_collision_material(prim_path, self._collision_material_path)

        # track observable joints of robot
        for obs_joint in robot.observable_joints:
            if obs_joint not in self._get_robot_joint_names(obj):
                raise f"Robot {robot.name} has no observable joint called {obs_joint}!"
            
            # append to observable obstacles: No environment offset will be applied
            self._observable_robot_joints.append(Articulation(prim_path + f"/{obs_joint}", f"env{env_idx}-{robot.name}/{obs_joint}"))

        # add reference to robot scene to current stage
        return prim_path

    def _add_collision_material(self, prim_path, material_path:str):
        # get prim path object
        prim = self._stage.GetPrimAtPath(prim_path)

        # add material
        from omni.physx.scripts.physicsUtils import add_physics_material_to_prim, PhysxSchema
        add_physics_material_to_prim(self._stage, prim, material_path)

        # register contact report api to forward collisions to _on_contact_report_event
        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contactReportAPI.CreateThresholdAttr().Set(200000)

    def _move_prim(self, path_from: str, path_to: str):
        """
        Moves the prim (path of object in simulation) from path_from to path_to.
        Returns the new path of the prim.
        """
        from omni.kit.commands import execute
        success, _ = execute("MovePrim", path_from=path_from, path_to=path_to)

        assert success == True, f"Failed to move prim from {path_from} to {path_to}"

        return path_to

    def _spawn_obstacle(self, obstacle: Obstacle, env_idx: int) -> str:
        prim_path = f"/World/env{env_idx}/{obstacle.name}"
        name = f"env{env_idx}-{obstacle.name}"

        # parse required class
        from omni.isaac.core.objects import FixedCuboid, DynamicCuboid, FixedSphere, DynamicSphere, DynamicCylinder, FixedCylinder
        from scripts.envs.isaac.random_dynamic_obstacles import RandomDynamicCuboid, RandomDynamicSphere, RandomDynamicCylinder
        from scripts.envs.isaac.random_static_obstacles import RandomFixedCuboid, RandomFixedSphere, RandomFixedCylinder

        # select corresponding class: Obstacle type, isRandomized?, isStatic?
        class_selector = {
            # select cubes
            (Cube, False, False): DynamicCuboid,
            (Cube, False, True): FixedCuboid,
            (Cube, True, False): RandomDynamicCuboid,
            (Cube, True, True): RandomFixedCuboid,
            # select spheres
            (Sphere, False, False): DynamicSphere,
            (Sphere, False, True): FixedSphere,
            (Sphere, True, False): RandomDynamicSphere,
            (Sphere, True, True): RandomFixedSphere,
            # select cylinders
            (Cylinder, False, False): DynamicCylinder,
            (Cylinder, False, True): FixedCylinder,
            (Cylinder, True, False): RandomDynamicCylinder,
            (Cylinder, True, True): RandomFixedCylinder
        }

        # parse equivalent of selected class in Isaac
        selected_class = class_selector.get(((type(obstacle)), obstacle.is_randomized(), obstacle.static), None)
        
        if selected_class is None:
            raise Exception(f"Obstacle of type {type(obstacle)}, random={obstacle.is_randomized()}, static={obstacle.static} isn't implemented!")

        # create parameter dict
        params = obstacle.get_constructor_params()
        params["prim_path"] = prim_path
        params["name"] = name

        # add env offset to position
        params["position"] = _add_position_offset(params["position"], self._env_offsets[env_idx])

        # create instance
        obstacle_obj = selected_class(**params)

        # add obstacle to scene
        self._scene.add(obstacle_obj)

        # track spawned obstacle
        self._obstacles.append((obstacle_obj, obstacle))

        # add it to list of observable objects, if necessary
        if obstacle.observable:
            self._observable_obstacles.append(obstacle_obj)

        # configure collision
        if obstacle.collision:
            # add collision material, allowing callbacks to register collisions in simulation
            self._add_collision_material(prim_path, self._collision_material_path)
        else:
            obstacle_obj.set_collision_enabled(False)

        return prim_path

    def _get_robot_joint_names(self, robot_articulation) -> List[str]:
        return [child.GetName() for child in robot_articulation.prim.GetAllChildren()]

    # static utillity functions
    @staticmethod
    def to_isaac_vector(vec3: np.ndarray):
        from pxr import Gf
        return Gf.Vec3f(list(vec3))

    @staticmethod
    def to_issac_quat(vec3: np.ndarray):
        from pxr import Gf
        a, b, c, d = list(vec3)
        return Gf.Quatf(float(a), float(b), float(c), float(d))

    @staticmethod
    def to_isaac_color(color: List[float]) -> np.ndarray:
        """
        Transform colour format into format Isaac accepts, ignoring opacity
        """
        return np.array(color[:-1])