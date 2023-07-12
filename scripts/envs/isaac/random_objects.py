from typing import Optional, Tuple
import numpy as np
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import FixedCuboid

class RandomCuboid(FixedCuboid):
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
        FixedCuboid.__init__(prim_path, name, position[0], translation[0], orientation[0], scale[0], visible, color, size, visual_material, physics_material)
        
        # save max random values
        self.max_pos = position[1]
        self.max_orientation = orientation[1]
        self.scale = scale

    def post_reset(self) -> None:
        pos, ori = self.get_default_state()

        # generate random numbers to allow setting random properties efficiently
        rand_floats = np.random.random_sample(3)

        # set random position and orientation  # todo: this exceeds max value
        self.set_world_pose(pos + self.max_pos * rand_floats[0], ori + self.max_orientation[1] * rand_floats[1])
        
        # set random scale   # todo: this exceeds max value
        self.set_local_scale(self.scale[0] + self.scale[1] * rand_floats[2])