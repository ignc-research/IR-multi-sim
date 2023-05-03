import math
from typing import List, Tuple
from scripts.envs.modular_env import ModularEnv
from scripts.rewards.distance import Distance, calc_distance
from scripts.spawnables.obstacle import Obstacle, Cube, Sphere, Cylinder
from scripts.spawnables.robot import Robot
from scripts.rewards.reward import Reward
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import *
from pathlib import Path

# from omni.isaac.core.tasks import BaseTask

class IsaacEnv(ModularEnv):
    def __init__(self, asset_path: str, step_size: float, headless: bool, robots: List[Robot], obstacles: List[Obstacle], rewards: List[Reward], num_envs: int, offset: Tuple[float, float]) -> None:
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
        self.asset_path = Path().absolute().joinpath(asset_path)

        # setup basic information about simulation
        self.num_envs = num_envs
        self.robots_per_env = len(robots)
        self.objs_per_env = self.robots_per_env + len(obstacles)

        # setup ISAAC simulation environment and interfaces
        self._setup_simulation(headless, step_size)
        self._setup_urdf_import()
        self._setup_physics()

        # setup rl environment
        self._setup_environments(robots, obstacles, [], offset)
        self._setup_observations(robots, obstacles)
        self._setup_rewards(rewards)
        
        # track robots and spawned objects
        self._setup_object_tracking()        

        # todo: fix robots not being imported correctly
        while True:
            self._simulation.update()

        # init bace class last, allowing it to automatically determine action and observation space
        super().__init__(step_size, headless, num_envs)
    
    def _setup_simulation(self, headless: bool, step_size: float):
        # isaac imports may only be used after SimulationApp is started (ISAAC uses runtime plugin system)
        from omni.isaac.kit import SimulationApp
        self._simulation = SimulationApp({"headless": headless})

        # make sure simulation was started
        assert self._simulation is not None, "Isaac Sim failed to start!"
        assert self._simulation.is_running, "Isaac Sim failed to start!"

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

    def _setup_environments(self, robots: List[Robot], obstacles: List[Obstacle], sensors: List, offset: Tuple[float, float]) -> None:
        # when to break to a new line in grid pattern
        break_index = math.ceil(math.sqrt(self.num_envs))

        # spawn objects for each environment
        for env_idx in range(self.num_envs):
            # calculate position offset for environment, creating grid pattern
            env_offset = np.array([(env_idx % break_index) * offset[0], math.floor(env_idx / break_index) * offset[1], 0])
            
            # create env root prim. Base objects automatically have their global position adjusted
            from omni.isaac.core.utils.prims import create_prim
            create_prim(f"/World/Env{env_idx}", position=env_offset)

            # spawn robots
            for robot in robots:
                # import robot from urdf, creating prim path
                prim_path = self._import_urdf(robot, f"/World/Env{env_idx}/Robots/{robot.name}")

                # modify prim path to match formating
                # prim_path = self._move_prim(prim_path, f"/World/Env{env_idx}/Robots/{robot.name}")

                while True:
                    self._simulation.update()

                # move robot to desired location
                from omni.isaac.core.articulations import Articulation
                obj = Articulation(prim_path, f"env{env_idx}-{robot.name}")
                self._scene.add(obj)
                
                obj.set_world_pose(robot.position, robot.orientation)

                # configure collision
                if robot.collision:
                    self._add_collision_material(prim_path, self._collision_material_path)

                print("Spawned", robot, prim_path)

            # spawn obstacles
            for obstacle in obstacles:
                prim_path = f"/World/Env{env_idx}/Obstacles/{obstacle.name}"

                if isinstance(obstacle, Cube):
                    self._create_cube(prim_path, obstacle.position, obstacle.orientation, obstacle.mass, obstacle.scale, obstacle.color, obstacle.collision)
                elif isinstance(obstacle, Sphere):
                    self._create_sphere(prim_path, obstacle.position, obstacle.mass, obstacle.radius, obstacle.color, obstacle.collision)
                elif isinstance(obstacle, Cylinder):
                    self._create_cylinder(prim_path, obstacle.position, obstacle.orientation, obstacle.mass, obstacle.radius, obstacle.height, obstacle.color, obstacle.collision)
                else:
                    raise f"Obstacle {type(obstacle)} implemented"
                
                print("Spawned", obstacle, prim_path)
                
            # spawn sensors
            for i, sensor in enumerate(sensors):
                raise "Sensors are not implemented"
            
    def _setup_object_tracking(self):
        # track spawned robots/obstacles/sensors
        from omni.isaac.core.articulations import ArticulationView
        self._robots = ArticulationView("/World/Env*/Robots/*", "Robots")
        self._scene.add(self._robots)
        # self._objects = ArticulationView("World/Env*/*", "Objects")
        # self._sensors = []  # todo: implement sensors

    def _setup_observations(self, robots: List[Robot], obstacles: List[Obstacle]) -> None:
        observable_paths = []

        # get observable objects for each environment
        for env_idx in range(self.num_envs):
            for robot in robots:
                # add robot position and rotation to list of observable objects
                if robot.observable:
                    observable_paths.append(f"/World/Env{env_idx}/Robots/{robot.name}")

                # add observable joints to list of observable objects
                for joint_name in robot.observable_joints:
                    observable_paths.append(f"/World/Env{env_idx}/Robots/{robot.name}/{joint_name}")
            
            # add observable obstacles to list
            for obstacle in obstacles:
                if obstacle.observable:
                    observable_paths.append(f"/World/Env{env_idx}/Obstacles/{obstacle.name}")

        # wrap observable objects in articulations, allowing to access their values
        from omni.isaac.core.articulations import Articulation
        self._observable_objects = [Articulation(prim_path, prim_path.split("/")[-1]) for prim_path in observable_paths]
        
    def _setup_rewards(self, rewards: List[Reward]) -> None:
        self.reward_fns = []

        for reward in rewards:
            if isinstance(reward, Distance):
                self.reward_fns.append(self._parse_distance_reward(reward))
            else:
                raise f"Reward {type(reward)} not implemented!"
        
    def _parse_distance_reward(self, distance: Distance):
        # parse indices in observations
        index_0 = self._find_observable_object(distance.obj1)
        index_1 = self._find_observable_object(distance.obj2)

        # parse function calculating distance to all targets
        def sum_distance() -> float:
            distance = 0
            # calculate the distance of all instances of the two objects in each env
            for i in range(0, self.num_envs * self.objs_per_env, self.objs_per_env):
                distance += calc_distance(self._obs[index_0 + i], self._obs[index_1 + i])
            return distance
                
        # minimize reward output
        if distance.minimize:
            def distance_reward():
                return sum_distance() * -1
            return distance_reward
        # maximize reward output
        else:
            return sum_distance
        
    def _find_observable_object(self, name: str) -> int:
        """
        Given the name of an observable object, tries to retrieve its position and rotation index in the observations.
        """
        for index, obj in enumerate(self._observable_objects):
            # todo: this only works for robot joints, not robots themselves
            if obj.prim_path.endswith(name):
                return index
        raise f"Object {name} must be observable if used for reward"

    def step_async(self, actions: np.ndarray) -> None:
        # apply actions to robots
        self._robots.set_joint_velocities(actions)

        # step once with simulations
        self._simulation.update()
    
    def step_wait(self) -> VecEnvStepReturn:
        # get observations
        self._obs = self._get_observations()

        # get rewards

        # get dones

        # get info

        raise "Not implemented"

    def reset(self) -> VecEnvObs:
        self._world.reset()

        # get observations from world
        self._obs = self._get_observations()

        return self._obs

    def get_robot_dof_limits(self) -> np.ndarray:
        # todo: ony get dof limits from robots of first environment
        print(self._robots.prims)
        print("Limits:", self._robots.get_dof_limits())
        raise "Not implemented!"
    
    def close(self) -> None:
        self._simulation.close()

    def _get_observations(self):
        return [obj.get_local_pose() for obj in self._observable_objects]

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

    def _import_urdf(self, robot: Robot, target_path: str) -> str:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique str identifying the newly spawned object within the engine.
        The is_robot flag determines whether the engine handles this object as a robot (something with movable links/joints) or a simple geometry object (a singular mesh).
        """
        abs_path = str(self._get_absolute_asset_path(robot.urdf_path))
        dest_path = f"./data/scenes/{robot.name}.usd"

        # import URDF to temporary scene
        from omni.kit.commands import execute
        success, prim_path = execute(
            "URDFParseAndImportFile", 
            urdf_path=abs_path, 
            import_config=self._config,
            dest_path=dest_path
        )

        # make sure import succeeded
        assert success, "Failed urdf import of: " + abs_path

        # add reference to robot scene to current stage
        robot_prim = self._stage.OverridePrim(target_path)
        robot_prim.GetReferences().AddReference(dest_path, prim_path)

        return target_path

    def _add_collision_material(self, prim_path: str, material_path:str):
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

    def _create_cube(
        self, 
        prim_path: str,
        position: np.ndarray,
        orientation: np.ndarray,
        mass: float,
        scale: List[float],
        color: List[float],
        collision: bool
        ) -> None:
        from omni.physx.scripts.physicsUtils import add_rigid_box

        # create cube
        add_rigid_box(
            self._stage, prim_path,
            size=self.to_isaac_vector(scale),
            position=self.to_isaac_vector(position),
            orientation=self.to_issac_quat(orientation),
            color=self.to_isaac_color(color),
            density=mass
        )

        if collision:
            self._add_collision_material(prim_path, self._collision_material_path)

    def _create_sphere(
        self,
        prim_path: str,
        position: np.ndarray,
        mass: float,
        radius: float,
        color: List[float],
        collision: bool
        ) -> None:
        from omni.physx.scripts.physicsUtils import add_rigid_sphere

        # create sphere
        add_rigid_sphere(
            self._stage, prim_path,
            radius=radius,
            position=self.to_isaac_vector(position),
            color=self.to_isaac_color(color),
            density=mass                
        )

        if collision:
            self._add_collision_material(prim_path, self._collision_material_path)
    
    def _create_cylinder(
        self,
        prim_path: str,
        position: np.ndarray,
        orientation: np.ndarray,
        mass: float,
        radius: float,
        height:float,
        color: List[float],
        collision: bool
    ) -> None:
        from omni.physx.scripts.physicsUtils import add_rigid_cylinder

        # create cylinder
        add_rigid_cylinder(
            self._stage, prim_path,
            radius=radius,
            height=height,
            position=self.to_isaac_vector(position),
            orientation=self.to_isaac_vector(orientation),
            color=self.to_isaac_color(color),
            density=mass
        )

        if collision:
            self._add_collision_material(prim_path, self._collision_material_path)

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