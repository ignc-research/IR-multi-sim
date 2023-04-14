from typing import List, Tuple, Union, Optional
from engines.engine import Engine
from rewards.distance import Distance, distance
from spawnables.obstacle import *
from modular_env import ModularEnv
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import *

class IsaacEngine(Engine):
    def __init__(self, asset_path: str, step_size: float, headless: bool = True) -> None:
        super().__init__(asset_path, step_size, headless)

        # isaac imports may only be used after SimulationApp is started (ISAAC uses runtime plugin system)
        from omni.isaac.kit import SimulationApp
        self._simulation = SimulationApp({"headless": headless})

        # make sure simulation was started
        assert self._simulation != None, "Isaac Sim failed to start!"

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

        # setup physics
        # subscribe to physics contact report event, this callback issued after each simulation step
        from omni.physx import get_physx_simulation_interface
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

        # track collisions
        self._collisions: List[Tuple(int, int)] = []

        # configure physics simulation
        from omni.physx.scripts.physicsUtils import UsdPhysics, UsdShade, Gf
        scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)

        # Configure default floor material
        self._floor_material_path = "/floorMaterial"
        UsdShade.Material.Define(self.stage, self._floor_material_path)
        floor_material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self._floor_material_path))
        floor_material.CreateStaticFrictionAttr().Set(0.0)
        floor_material.CreateDynamicFrictionAttr().Set(0.0)
        floor_material.CreateRestitutionAttr().Set(1.0)

        # Configure default collision material
        self._collision_material_path = "/collisionMaterial"
        UsdShade.Material.Define(self.stage, self._collision_material_path)
        material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self._collision_material_path))
        material.CreateStaticFrictionAttr().Set(0.5)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0.9)
        material.CreateDensityAttr().Set(0.001) 

        # setup ground plane
        ground_prim_path = "/World/defaultGroundPlane"
        self._scene.add_default_ground_plane(prim_path=ground_prim_path)

        # add collision to ground plane
        self._add_collision_material(ground_prim_path, self._floor_material_path)

        # track spawned robots/obstacles/sensors
        from omni.isaac.core.articulations import ArticulationView
        self._robots = ArticulationView("World/Env*/Robots/*", "Robots")
        self._objects = ArticulationView("World/Env*/*", "Objects")
        self._sensors = []  # todo: implement sensors
        
        # create buffers for observations, rewards, done, info
        self._obs: VecEnvObs = None
        self._rewards: np.ndarray = None
        self._done: np.ndarray = None
        self._info: List[Dict] = None
        
    
    def set_up(self, env: ModularEnv) -> Tuple[List[int], List[int], List[int]]:
        # setup each environment
        self._setup_environments(env)
        
        # setup efficent method to track observable objects
        self._setup_observations(env)

        # setup efficient method to get rewards
        self._setup_rewards(env)

        # reset world to allow physics object to interact
        self._world.reset()

    def _setup_environments(self, env: ModularEnv) -> None:
        for env_id in env.num_envs:
            # calculate position offset for environment, creating grid pattern
            pos_offset = (env_id % env.num_envs) * env.offset[0], (env_id / env.num_envs) * env.offset[1]

            # spawn robots
            for robot in env.robots:
                # import robot from urdf, creating prim path
                prim_path = self._import_urdf(robot.urdf_path)

                # modify prim path to match formating
                prim_path = self._move_prim(prim_path, f"World/Env{env_id}/Robots/{robot.name}")

                # configure collision
                if robot.collision:
                    self._add_collision_material(prim_path, self._collision_material_path)

                # move robot to desired location
                from omni.isaac.core.articulations import Articulation
                obj = Articulation(prim_path)
                obj.set_world_pose(robot.position + pos_offset, robot.orientation)

            # spawn obstacles
            for obstacle in env.obstacles:
                prim_path = f"World/Env{env_id}/Obstacles/{obstacle.name}"
                if isinstance(obstacle, Cube):
                    self._create_cube(prim_path, pos_offset, **dir(obstacle))
                elif isinstance(obstacle, Sphere):
                    self._create_sphere(prim_path, pos_offset, **dir(obstacle))
                elif isinstance(obstacle, Cylinder):
                    self._create_cylinder(prim_path, pos_offset, **dir(obstacle))
                else:
                    raise f"Obstacle {type(obstacle)} implemented"
                
            # spawn sensors
            for i, sensor in enumerate(env.sensors):
                raise "Sensors are not implemented"

    def _setup_observations(self, env: ModularEnv) -> None:
        # setup articulations for robot observations
        from omni.isaac.core.articulations import ArticulationView
        # for each environment, for each robot, allow retrieving their local pose
        observable_robots = "|".join([f"Robots/{robot.name}" for robot in env.robots if robot.observable])

        # for each environment, for each robot, select their observable joints and allow querying their values
        observable_joints = "|".join([f"Robots/{robot.name}/{robot.observable_joints}" for robot in env.robots])

        # for each environment, for each obstacle, allow retrieving their local pose
        observable_obstacles = "|".join([f"Obstacles/{obstacle.name}" for obstacle in env.obstacles if obstacle.observable])

        # create main regex: join observable objects
        regex = "|".join([observable_robots, observable_joints, observable_obstacles])
        regex = f"World/Env*/({regex})"
        self._observations = ArticulationView(regex, "Observations")

    def _setup_rewards(self, env: ModularEnv) -> None:
        self.reward_fns = []

        for reward in env.rewards:
            if isinstance(reward, Distance):
                self.reward_fns.append(self._parse_distance_reward(reward))
            else:
                raise f"Reward {type(reward)} not implemented!"


    def _parse_distance_reward(self, distance: Distance):
        # get indices of observable objects
        index_0 = self._get_obs_obj_index(distance.obj1)
        index_1 = self._get_obs_obj_index(distance.obj2)

        def reward() -> float:
            obs = self._obs

            # todo: get observations, calculate distance between objects in all environments            
            raise "Not implemented!"

        return reward

    def _get_obs_obj_index(self, name: str) -> int:
        """
        Tries to retrieve an objects index in the observation array with given name from the list of observed objects.
        """
        for i, prim_path in self._observations.body_names:
            if prim_path.endswith(name):
                return i
        
        raise f"Object with name {name} isn't an observed object!"

    def set_joint_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint positions of all robots specified in robot_indices to their respective values specified in positions.
        """
        self._robots.set_joint_positions(positions, robot_indices, joint_indices)
    
    def set_joint_position_targets(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint position targets of all robots specified in robot_indices to their respective values specified in positions.
        """
        self._robots.set_joint_position_targets(positions, robot_indices, joint_indices)

    def set_joint_velocities(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor]],
        robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint velocities of all robots specified in robot_indices to their respective values specified in velocities.
        """
        self._robots.set_joint_velocities(velocities, robot_indices, joint_indices)
    
    def set_joint_velocity_targets(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor]],
        robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint velocities targets of all robots specified in robot_indices to their respective values specified in velocities.
        """
        self._robots.set_joint_velocity_targets(velocities, robot_indices, joint_indices)

    def set_local_poses(
        self,
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        self._objects.set_local_poses(translations, orientations, indices)

    def get_local_poses(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        return self._objects.get_local_poses(indices)

    def get_sensor_data(
        self, 
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> List[List]:
        """
        Gets the sensor data generated by all sensors.
        """
        raise "Not implemented!"

    def get_collisions(self) -> List[Tuple[int, int]]:
        """
        Returns the ids of objects which are colliding. Updated after each step.
        Example: [(1, 2), (1, 3)] -> Object 1 is colliding with object 2 and 3.
        """
        return self._collisions

    def step(self, actions) -> VecEnvStepReturn:
        """
        Steps the environment for one timestep
        """
        # apply actions # todo: set control mode for all robots with enum
        self._robots.set_joint_position_targets(actions)

        self._simulation.update()

        # get rewards

        # get dones

        # get info

        raise "Not implemented"

    def reset(self) -> VecEnvObs:
        self._world.reset()

        # todo: format
        self._observations.get_local_poses()

        raise "Not implemented"

    def get_robot_dof_limits(self) -> Union[np.ndarray, torch.Tensor]:
        # todo: ony get dof limits from robots of first environment
        self._robots.get_dof_limits()
        raise "Not implemented!"

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

    def _import_urdf(self, urdf_path: str) -> str:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique str identifying the newly spawned object within the engine.
        The is_robot flag determines whether the engine handles this object as a robot (something with movable links/joints) or a simple geometry object (a singular mesh).
        """
        abs_path = self._get_absolute_asset_path(urdf_path)

        # import URDF
        from omni.kit.commands import execute
        success, prim_path = execute("URDFParseAndImportFile", urdf_path=abs_path, import_config=self._config)

        # make sure import succeeded
        assert success, "Failed urdf import of: " + abs_path

        return prim_path

    def _add_collision_material(self, prim_path: str, material_path:str):
        # add material
        from omni.physx.scripts.physicsUtils import add_physics_material_to_prim, PhysxSchema
        add_physics_material_to_prim(self._stage, prim_path, material_path)

        # register contact report api to forward collisions to _on_contact_report_event
        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(prim_path))
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
        offset: np.ndarray,
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
            self.stage, prim_path,
            size=to_isaac_vector(scale),
            position=to_isaac_vector(position + offset),
            orientation=to_issac_quat(orientation),
            color=to_isaac_color(color),
            density=mass
        )

        if collision:
            self._add_collision_material(prim_path, self._collision_material_path)

    def _create_sphere(
        self,
        prim_path: str,
        position: np.ndarray,
        offset: np.ndarray,
        mass: float,
        radius: float,
        color: List[float],
        collision: bool
        ) -> None:
        from omni.physx.scripts.physicsUtils import add_rigid_sphere

        # create sphere
        add_rigid_sphere(
            self.stage, prim_path,
            radius=radius,
            position=to_isaac_vector(position + offset),
            color=to_isaac_color(color),
            density=mass                
        )

        if collision:
            self._add_collision_material(prim_path, self._collision_material_path)
    
    def _create_cylinder(
        self,
        prim_path: str,
        position: np.ndarray,
        offset: np.ndarray,
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
            self.stage, prim_path,
            radius=radius,
            height=height,
            position=self.to_isaac_vector(position + offset),
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