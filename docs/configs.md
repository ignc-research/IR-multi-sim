# Documentation of the spawnable objects and usable settings

## URDF
- Table

## Robots
Robots need to be defined in an [urdf fiile](http://wiki.ros.org/urdf). The file needs to be saved in the "./data/robots" directory.

Example:
``` python
from scripts.spawnables.robot import Robot

r1 = Robot("robots/ur5/urdf/ur5_with_gripper.urdf",  # relative path to urdf file from data directory
    position=np.array([0, 1, 2]),                    # x,y,z coordinates from environment origin
    orientation=np.array([1, 0, 0, 0]),              # orientation in quaternion
    observalbe_joints=["ee_link"]                    # mark end effector ling as observable
)
```

## Objects
Three types of obstacles are currently supported: Cubes, Speheres and Cylinders. Colour, collision, observability and more can be configured with the parameters of the obstacle. Per default, all obstacles are observable.

Example:
``` python
from scripts.spawnables.obstacle import Cube

target_cube = Cube(
    position=np.array([0.4, 1, 2]),  # relative x,y,z coordinates from environment origin
    color=np.array([0, 1, 0]),       # color of cube, in RGB values.
    scale=[0.1, 0.8, 1.3],           # scale of cube
    name="TargetCube"                # name of cube, allowing it to be referenced by reward functions
)
```

## Rewards
Rewards are functions which evluate the current environment state, rating desirable states with a high value and undesirable states with a low value.

Example:
``` python
from scripts.rewards.distance import Distance

# create distance function using names of previously spawned objects
target_distance = Distance(
    obj1="TargetCube",              # Reference the target cube by name
    obj2="R1/ee_link",              # Reference the ee_link of R1 robot by name (must be observable)
    name="TargetDistance",          # Name of the distance, allowing it to be referenced by reset functions
    minimize=True                   # True if the distance is supposed to be minimized, otherwise false
)

# create distance function using the previously spawned object
target_distance = Distance(
    obj_1=target_cube,
    obj_2="R1/ee_link",
    name="TargetDistance",
    minimize=True
)
```

## Resets
Resets are functions which determine wether the environment needs to be reset. This included exceeded min or max values of a previously defined distance function, or an exceeded number of timesteps.

``` python
from scripts.resets.distance_reset import DistanceReset

distance_reset = DistanceReset(
    distance=target_distance,       # Refefence can also be passed as name ("TargetDistance")
    min=0.1,                        # If distance drops below 0.1: Reset environment
    max=1.5                         # If distance exceeds 1.5: Reset environment
)
