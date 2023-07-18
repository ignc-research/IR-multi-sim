import math
from typing import List, Tuple
from scripts.envs.modular_env import ModularEnv
from scripts.envs.params.env_params import EnvParams
from scripts.rewards.distance import Distance, calc_distance
from scripts.spawnables.obstacle import Obstacle, Cube, Sphere, Cylinder
from scripts.spawnables.random_obstacle import RandomCube, RandomSphere, RandomCylinder
from scripts.spawnables.robot import Robot
from scripts.rewards.reward import Reward
from scripts.resets.reset import Reset
from scripts.resets.distance_reset import DistanceReset
from scripts.resets.timesteps_reset import TimestepsReset
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import *
from pathlib import Path


def _add_offset_to_tuple(tuple: Tuple[float, float], offset: float) -> Tuple[float, float]:
    return (tuple[0] + offset, tuple[1] + offset)


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

        # save the distances in the current environment
        self._distances: Dict[str, List[float]] = {}

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
        super().__init__(params.step_size, params.headless, params.num_envs)
    
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
        self._world = World(physics_dt=step_size)
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
        self._config.convex_decomp = False
        self._config.import_inertia_tensor = True
        self._config.fix_base = True
        self._config.create_physics_scene = True

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
        scene.CreateGravityMagnitudeAttr().Set(981.0)

        # Configure default floor material
        self._floor_material_path = "/floorMaterial"
        UsdShade.Material.Define(self._stage, self._floor_material_path)
        floor_material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._floor_material_path))
        floor_material.CreateStaticFrictionAttr().Set(0.0)
        floor_material.CreateDynamicFrictionAttr().Set(0.0)
        floor_material.CreateRestitutionAttr().Set(1.0)

        # Configure default collision material
        self._collision_material_path = "/collisionMaterial"
        UsdShade.Material.Define(self._stage, self._collision_material_path)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._collision_material_path))
        material.CreateStaticFrictionAttr().Set(0.5)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0.9)
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
                if isinstance(obstacle, Cube):
                    self._spawn_cube(obstacle, env_idx)
                elif isinstance(obstacle, RandomCube):
                    self._spawn_random_cube(obstacle, env_idx)
                elif isinstance(obstacle, Sphere):
                    self._spawn_sphere(obstacle, env_idx)
                elif isinstance(obstacle, Cylinder):
                    self._spawn_cylinder(obstacle, env_idx)
                else:
                    raise f"Obstacle {type(obstacle)} not implemented"
        
    def _setup_rewards(self, rewards: List[Reward]) -> None:
        self._reward_fns = []

        for reward in rewards:
            if isinstance(reward, Distance):
                self._reward_fns.append(self._parse_distance_reward(reward))
            else:
                raise f"Reward {type(reward)} not implemented!"
        
    def _parse_distance_reward(self, distance: Distance):
        # parse indices in observations
        obj1_start, _ = self._parse_observable_object_range(distance.obj1)
        obj2_start, _ = self._parse_observable_object_range(distance.obj2)

        # extract name to allow created function to access it easily
        name = distance.name

        # parse function calculating distance to all targets
        def distance_per_env() -> np.ndarray:
            # calculate distances as np array
            result = []
            for i in range(self.num_envs):
                result.append(calc_distance(
                    self._obs[i][obj1_start:obj1_start+3],
                    self._obs[i][obj2_start:obj2_start+3]
                ))
            result = np.array(result)

            # save distance for current iteration
            self._distances[name] = result

            return result
    
        # minimize reward output
        if distance.minimize:
            def distance_reward():
                return distance_per_env() * [-1 for _ in range(self.num_envs)]
            return distance_reward
        # maximize reward output
        else:
            return distance_per_env
    
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

        # apply actions to robots
        for i, robot in enumerate(self._robots):
            robot.set_joint_positions(actions[i])

        # step simulation amount of times according to params
        for _ in range(self.step_count):
            self._simulation.update()
    
    def step_wait(self) -> VecEnvStepReturn:
        # get observations
        self._obs = self._get_observations()

        # get rewards
        self._rewards = self._get_rewards()

        # get dones
        self._dones = self._get_dones()

        # print("Obs    :", self._obs)
        # print("Rewards:", self._rewards)
        # print("Dones  :", self._dones)
        # print("Timest.:", self._timesteps)

        return self._obs, self._rewards, self._dones, self.env_data

    def reset(self, env_idxs: np.ndarray=None) -> VecEnvObs:
        # reset entire simulation
        if env_idxs is None:
            # reset the world
            self._world.reset()

            # reset timestep tracking
            self._timesteps = np.zeros(self.num_envs)

            # reset observations
            self._obs = self._get_observations()

        # reset envs manually
        else:
            # reset timestep tracking
            self._timesteps[env_idxs] = 0

            # select each environment
            for i in env_idxs:
                # reset all obstacles to default pose
                for geometryPrim, obstacle in self._get_obstacles(i):
                    geometryPrim.post_reset()

                # reset all robots to default pose
                for robot in self._get_robots(i):
                    robot.post_reset()
        
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
            joint_idx_offset = self.observable_robot_joint_count
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

    def _spawn_cube(self, cube: Cube, env_idx: int) -> str:
        prim_path = f"/World/env{env_idx}/{cube.name}"
        name = f"env{env_idx}-{cube.name}"

        # create cube
        from omni.isaac.core.objects import FixedCuboid
        cube_obj = FixedCuboid(
            prim_path,
            name,
            cube.position + self._env_offsets[env_idx],
            None,
            cube.orientation,
            cube.scale,
            color=cube.color
        )
        self._scene.add(cube_obj)

        # track spawned cube
        self._obstacles.append((cube_obj, cube))

        # add it to list of observable objects, if necessary
        if cube.observable:
            self._observable_obstacles.append(cube_obj)

        # configure collision
        if cube.collision:
            # add collision material, allowing callbacks to register collisions in simulation
            self._add_collision_material(prim_path, self._collision_material_path)
        else:
            cube_obj.set_collision_enabled(False)

        return prim_path

    def _spawn_random_cube(self, cube: RandomCube, env_idx: int) -> str:
        prim_path = f"/World/env{env_idx}/{cube.name}"
        name = f"env{env_idx}-{cube.name}"

        # create cube
        from scripts.envs.isaac.random_objects import RandomCuboid
        cube_obj = RandomCuboid(
            prim_path,
            _add_offset_to_tuple(cube.position, self._env_offsets[env_idx]),
            cube.orientation,
            cube.scale,
            name,
            color=cube.color
        )
        self._scene.add(cube_obj)

        # track spawned cube
        self._obstacles.append((cube_obj, cube))

        # add it to list of observable objects, if necessary
        if cube.observable:
            self._observable_obstacles.append(cube_obj)

        # configure collision
        if cube.collision:
            # add collision material, allowing callbacks to register collisions in simulation
            self._add_collision_material(prim_path, self._collision_material_path)
        else:
            cube_obj.set_collision_enabled(False)

        return prim_path

    def _spawn_sphere(self, sphere: Sphere, env_idx: int) -> str:
        prim_path = f"/World/env{env_idx}/{sphere.name}"
        name = f"env{env_idx}-{sphere.name}"

        # create sphere
        from omni.isaac.core.objects import FixedSphere
        sphere_obj = FixedSphere(
            prim_path,
            name,
            sphere.position + self._env_offsets[env_idx],
            None,
            sphere.orientation,
            radius=sphere.radius,
            color=sphere.color
        )
        self._scene.add(sphere_obj)
    
        # track spawned sphere
        self._obstacles.append((sphere_obj, sphere))

        # add it to list of observable objects, if necessary
        if sphere.observable:
            self._observable_obstacles.append(sphere_obj)

        # configure collision
        if sphere.collision:
            # add collision material, allowing callbacks to register collisions in simulation
            self._add_collision_material(prim_path, self._collision_material_path)
        else:
            sphere_obj.set_collision_enabled(False)

        return prim_path
    
    def _spawn_cylinder(self, cylinder: Cylinder, env_idx:int) -> str:
        prim_path = f"/World/env{env_idx}/{cylinder.name}"
        name = f"env{env_idx}-{cylinder.name}"

        # create cylinder
        from omni.isaac.core.objects import FixedCylinder
        cylinder_obj = FixedCylinder(
            prim_path,
            name,
            cylinder.position + self._env_offsets[env_idx],
            None,
            cylinder.orientation,
            radius=cylinder.radius,
            height=cylinder.height,
            color=cylinder.color
        )
        self._scene.add(cylinder_obj)
    
        # track spawned cylinder
        self._obstacles.append((cylinder_obj, cylinder))

        # add it to list of observable objects, if necessary
        if cylinder.observable:
            self._observable_obstacles.append(cylinder_obj)

        # configure collision
        if cylinder.collision:
            # add collision material, allowing callbacks to register collisions in simulation
            self._add_collision_material(prim_path, self._collision_material_path)
        else:
            cylinder_obj.set_collision_enabled(False)

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