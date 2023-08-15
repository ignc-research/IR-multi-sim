from typing import Optional, Tuple, Sequence
import numpy as np
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import FixedCuboid, FixedSphere, FixedCylinder

def _get_value_in_range(min: float, max: float, range: float) -> float:
    """Returns a value between min and max.
        Example: Min=0, Max=10, range:0.5 -> 5

    Args:
        min (float): Min value
        max (float): Max value
        range (float): [0, 1]

    Returns:
        float: _description_
    """

    return min + (max - min) * range

class RandomFixedCuboid(FixedCuboid):
    def __init__(
        self,
        prim_path: str,
        position: Tuple[np.ndarray, np.ndarray],
        orientation: Tuple[np.ndarray, np.ndarray],
        scale: Tuple[np.ndarray, np.ndarray],
        name: str = "random_cube",
        translation: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        """A cube spawned in Issac Sim, randomizing position, orientation and scale at each reset

        Args:
            prim_path (str): _description_
            position (Tuple[np.ndarray, np.ndarray]): Min and max position. Shape of array is (3,)
            orientation (Tuple[np.ndarray, np.ndarray]): Min and max orieantation. Shape of array is (4,)
            scale (Tuple[np.ndarray, np.ndarray]): Min and max scale. Shape of array is (3,)
            name (str, optional): _description_. Defaults to "random_cube".
            translation (Optional[np.ndarray], optional): _description_. Defaults to None.
            visible (Optional[bool], optional): _description_. Defaults to None.
            color (Optional[np.ndarray], optional): _description_. Defaults to None.
            size (Optional[float], optional): _description_. Defaults to None.
            visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
            physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
        """
        
        # init base class with default lowest values
        FixedCuboid.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position[0],
            translation=translation,
            orientation=orientation[0],
            scale=scale[0],
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save max random values
        self.max_pos = position[1]
        self.max_orientation = orientation[1]
        self.scale = scale

    def post_reset(self) -> None:
        # get current position and orientation as XFormPrimState
        state = self.get_default_state()

        # generate random numbers to allow setting random properties efficiently
        rand_floats = np.random.random_sample(3)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(state.position, self.max_pos, rand_floats[0])
        ori = _get_value_in_range(state.orientation, self.max_orientation, rand_floats[1])
        scale = _get_value_in_range(self.scale[0], self.scale[1], rand_floats[2])

        # set random position and orientation
        self.set_world_pose(pos, ori)
        
        # set random scale
        self.set_local_scale(scale)

class RandomFixedSphere(FixedSphere):
    def __init__(
        self,
        prim_path: str,
        position: Tuple[np.ndarray, np.ndarray],
        orientation: Tuple[np.ndarray, np.ndarray],
        scale: Tuple[np.ndarray, np.ndarray],
        name: str = "random_cube",
        translation: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        """A sphere spawned in Issac Sim, randomizing position, orientation and scale at each reset

        Args:
            prim_path (str): _description_
            position (Tuple[np.ndarray, np.ndarray]): Min and max position. Shape of array is (3,)
            orientation (Tuple[np.ndarray, np.ndarray]): Min and max orieantation. Shape of array is (4,)
            scale (Tuple[np.ndarray, np.ndarray]): Min and max scale. Shape of array is (3,)
            name (str, optional): _description_. Defaults to "random_cube".
            translation (Optional[np.ndarray], optional): _description_. Defaults to None.
            visible (Optional[bool], optional): _description_. Defaults to None.
            color (Optional[np.ndarray], optional): _description_. Defaults to None.
            size (Optional[float], optional): _description_. Defaults to None.
            visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
            physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
        """
        

        # init base class with default lowest values
        FixedSphere.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position[0],
            translation=translation,
            orientation=orientation[0],
            scale=scale[0],
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save max random values
        self.max_pos = position[1]
        self.max_orientation = orientation[1]
        self.scale = scale

    def post_reset(self) -> None:
        # get current position and orientation as XFormPrimState
        state = self.get_default_state()

        # generate random numbers to allow setting random properties efficiently
        rand_floats = np.random.random_sample(3)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(state.position, self.max_pos, rand_floats[0])
        ori = _get_value_in_range(state.orientation, self.max_orientation, rand_floats[1])
        scale = _get_value_in_range(self.scale[0], self.scale[1], rand_floats[2])

        # set random position and orientation
        self.set_world_pose(pos, ori)
        
        # set random scale
        self.set_local_scale(scale)

class RandomFixedCylinder(FixedCylinder):
    def __init__(
        self,
        prim_path: str,
        position: Tuple[np.ndarray, np.ndarray],
        orientation: Tuple[np.ndarray, np.ndarray],
        scale: Tuple[np.ndarray, np.ndarray],
        name: str = "random_cube",
        translation: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        """A cube spawned in Issac Sim, randomizing position, orientation and scale at each reset

        Args:
            prim_path (str): _description_
            position (Tuple[np.ndarray, np.ndarray]): Min and max position. Shape of array is (3,)
            orientation (Tuple[np.ndarray, np.ndarray]): Min and max orieantation. Shape of array is (4,)
            scale (Tuple[np.ndarray, np.ndarray]): Min and max scale. Shape of array is (3,)
            name (str, optional): _description_. Defaults to "random_cube".
            translation (Optional[np.ndarray], optional): _description_. Defaults to None.
            visible (Optional[bool], optional): _description_. Defaults to None.
            color (Optional[np.ndarray], optional): _description_. Defaults to None.
            size (Optional[float], optional): _description_. Defaults to None.
            visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
            physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
        """
        

        # init base class with default lowest values
        FixedCylinder.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position[0],
            translation=translation,
            orientation=orientation[0],
            scale=scale[0],
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save max random values
        self.max_pos = position[1]
        self.max_orientation = orientation[1]
        self.scale = scale

    def post_reset(self) -> None:
        # get current position and orientation as XFormPrimState
        state = self.get_default_state()

        # generate random numbers to allow setting random properties efficiently
        rand_floats = np.random.random_sample(3)

        # generate new random position, orientation and scale
        pos = _get_value_in_range(state.position, self.max_pos, rand_floats[0])
        ori = _get_value_in_range(state.orientation, self.max_orientation, rand_floats[1])
        scale = _get_value_in_range(self.scale[0], self.scale[1], rand_floats[2])

        # set random position and orientation
        self.set_world_pose(pos, ori)
        
        # set random scale
        self.set_local_scale(scale)