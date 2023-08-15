from typing import Optional, Tuple, Sequence
import numpy as np
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCylinder

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

class RandomDynamicCuboid(DynamicCuboid):
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
            mass (Optional[float], optional): _description_. Defaults to None.
            density (Optional[float], optional): _description_. Defaults to None.
            linear_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
            angular_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
        """
        # todo: check if params are tuples or singular params -> en/disable randomization of parameter
        print(position, orientation, scale)
        
        # init base class with default lowest values
        DynamicCuboid.__init__(
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
            physics_material=physics_material,
            mass=mass,
            density=density,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
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
        
        # set random scale
        self.set_local_scale(scale)

        # obj with rigid bodies need to be reset by internal C++ callback
        self.set_default_state(pos, ori)
        DynamicCuboid.post_reset(self)

class RandomDynamicSphere(DynamicSphere):
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
        DynamicSphere.__init__(
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
        
        # set random scale
        self.set_local_scale(scale)

        # obj with rigid bodies need to be reset by internal C++ callback
        self.set_default_state(pos, ori)
        DynamicSphere.post_reset(self)

class RandomDynamicCylinder(DynamicCylinder):
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
        DynamicCylinder.__init__(
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
        
        # set random scale
        self.set_local_scale(scale)

        # obj with rigid bodies need to be reset by internal C++ callback
        self.set_default_state(pos, ori)
        DynamicCylinder.post_reset(self)