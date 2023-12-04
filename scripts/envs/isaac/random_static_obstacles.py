from typing import Optional, Tuple, Sequence
import numpy as np
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import FixedCuboid, FixedSphere, FixedCylinder

def get_value_in_range(min: np.array, max: np.array) -> np.array:
    """Returns a value between min and max.
        Example: Min=0, Max=10, range:0.5 -> 5

    Args:
        min (float): Min value
        max (float): Max value
        range (float): [0, 1]

    Returns:
        float: _description_
    """

    return min + (max - min) * np.random.random_sample(min.size)

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
        physics_material: Optional[PhysicsMaterial] = None
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
        
        # expand input parameters if no random values were specified
        if(type(position) is not tuple):
            position = (position, None)
        if(type(orientation) is not tuple):
            orientation = (orientation, None)
        if(type(scale) is not tuple):
            scale = (scale, None)

        # init base class with default lowest values
        FixedCuboid.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position[0],
            translation=translation,
            orientation=None,
            scale=scale[0],
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save min and max range
        self.position = position
        self.orientation = orientation
        self.scale = scale

    def post_reset(self) -> None:
        # generate new random position, orientation and scale (if necessary)
        pos_min, pos_max = self.position
        ori_min, ori_max = self.orientation
        scale_min, scale_max = self.scale

        # randomize position if necessary
        if pos_max is None:
            pos = pos_min
        else:
            pos = get_value_in_range(pos_min, pos_max)
        
        # randomize orientation if necessary
        if ori_max is None:
            ori = ori_min
        else:
            ori = get_value_in_range(ori_min, ori_max)

        # randomize scale if necessary
        if scale_max is not None:
            self.set_local_scale(get_value_in_range(scale_min, scale_max))

        # set random position and orientation
        self.set_world_pose(pos, ori)
        

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
        radius: Optional[float] = None,
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
        
        # expand input parameters if no random values were specified
        if(type(position) is not tuple):
            position = (position, None)
        if(type(orientation) is not tuple):
            orientation = (orientation, None)
        if(type(scale) is not tuple):
            scale = (scale, None)

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
            radius = radius,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save min and max range
        self.position = position
        self.orientation = orientation
        self.scale = scale

    def post_reset(self) -> None:
        # generate new random position, orientation and scale (if necessary)
        pos_min, pos_max = self.position
        ori_min, ori_max = self.orientation
        scale_min, scale_max = self.scale

        # randomize position if necessary
        if pos_max is None:
            pos = pos_min
        else:
            pos = get_value_in_range(pos_min, pos_max)
        
        # randomize orientation if necessary
        if ori_max is None:
            ori = ori_min
        else:
            ori = get_value_in_range(ori_min, ori_max)

        # randomize scale if necessary
        if scale_max is not None:
            self.set_local_scale(get_value_in_range(scale_min, scale_max))

        # set random position and orientation
        self.set_world_pose(pos, ori)

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
        radius: Optional[float] = None,
        height: Optional[float] = None,
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
        # expand input parameters if no random values were specified
        if(type(position) is not tuple):
            position = (position, None)
        if(type(orientation) is not tuple):
            orientation = (orientation, None)
        if(type(scale) is not tuple):
            scale = (scale, None)

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
            radius = radius,
            height = height,
            visual_material=visual_material,
            physics_material=physics_material
        )
        
        # save min and max range
        self.position = position
        self.orientation = orientation
        self.scale = scale

    def post_reset(self) -> None:
        # generate new random position, orientation and scale (if necessary)
        pos_min, pos_max = self.position
        ori_min, ori_max = self.orientation
        scale_min, scale_max = self.scale

        # randomize position if necessary
        if pos_max is None:
            pos = pos_min
        else:
            pos = get_value_in_range(pos_min, pos_max)
        
        # randomize orientation if necessary
        if ori_max is None:
            ori = ori_min
        else:
            ori = get_value_in_range(ori_min, ori_max)

        # randomize scale if necessary
        if scale_max is not None:
            self.set_local_scale(get_value_in_range(scale_min, scale_max))

        # set random position and orientation
        self.set_world_pose(pos, ori)