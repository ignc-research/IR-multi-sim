# Documentation of the spawnable objects and usable settings
The YAML settings file comprises several definable elements that focus on the runtime settings and the specific environment.

## General runtime parameter 
```yaml
  run:
  load_model:               # bool: Create a new model or utilize an existing one.
  model_name:               # str: Should be a string representing the model folder.
  checkpoint:               # str: Name of the model version for continuing training (including .zip).
  algorithm:                # Specify custom algorithm settings; default is used if not defined.
    type:                   # str: Choose from "PPO," "TD3," "SAC," "A2C," or "DDPG".
    parameters:             # Define various parameters for your model, e.g.:
      gamma:                # float: Reinforcement learning-specific parameter.
      learning_rate:        # float: Learning rate of the model.
      batch_size:           # int: Batch size for the model.
      ...
    custom_policy:          # Define specific layer settings and activation function.
      activation_function:  # str: Choose between "RELU" or "Tanh".
      value_function:       # array[int]: Define layer sizes.
        - ...
      policy_function:
        - ...
```

## Training and evaluation parameter
```yaml 
train:
  logging:      # int: Level of information output during learning [0, 1, 2, 3, 4].
  timesteps:    # int: Number of timesteps for the model to learn.
  save_freq:    # int: Frequency at which the model should be saved.

evaluation:
  timesteps:    # int: Number of timesteps used for model evaluation.
  logging:      # int: Level of information output during learning [0, 1, 2, 3, 4].
```

## Environment parameter
```yaml 
env:
  num_envs:     # int: Number of simultaneously running environments.
  env_offset:   # [int, int]: Size in the x and y directions of an environment.
  headless:     # bool: Run the simulation without visualization.
  step_size:    # float: Size of one simulation step; generally, 240Hz = 1/240.
  step_count:   # int: Number of steps the simulation can perform in each iteration.

```

### Objects within an environment
Multiple objects of the same type can be created as needed by starting a new object definition under the specific object type with a new hyphen. This project works with general observability that can be defined for all objects, rather than specific sensors. As a result, robots, robot joints, and obstacles can be marked as observable. This means that their relative position, orientation, and scale will be included in the observations of the machine learning model.

#### Robots
To create a robot in an environment, only the urdf path is required. All other parameters are optional. The robot must be defined in an [urdf](http://wiki.ros.org/urdf) file and saved in the "./data/robots" directory.
```yaml 
  robots:
  - name:                 # str: Name of the robot.
    urdf_path:            # str: Path to the URDF, usually in the robots folder.
    position:             # [float, float, float]: X, Y, Z base position of the robot.
    orientation:          # [float, float, float, float]: X, Y, Z, W world space quaternion of the robot.
    collision:            # bool: True if the robot is supposed to collide with surroundings.
    observable:           # bool: True if position and orientation are included in observations for training.
    observable_joints:    # ["string", ...]: Names of robot joints to be observed.
    control_type:         # str: "Velocity" or "Position" to define the control type.
    max_velocity:         # float: Define the maximum velocity a joint can be moved by.
    
  - name:                 # Start a new robot object.
```

#### General URDFs
To define a general URDF object, you must provide a path to a URDF file. The other parameters are optional.
```yaml 
  urdfs:
  - name:         # str: Name of the URDF.
    urdf_path:    # str: Path of the URDF.
    scale:        # [float, float, float]: Scaling along the x, y, z axes.
    position:     # [float, float, float]: X, Y, Z position.
    orientation:  # [float, float, float, float]: X, Y, Z, W world space quaternion.
    collision:    # bool: True if it is supposed to collide with robots.
    observable:   # bool: True if position and orientation are included in observations for training.
```  

#### Spawnable Objects
To define objects that can be spawned, you must provide the object type. The other parameters are optional. You can define the position, orientation, and scale for each object deterministically with a 3-dimensional list. Additionally, you can define ranges with minimum and maximum valid values (e.g. [[min, min, min], [max, max, max]]) to automatically create randomized values for each reset.
```yaml  
  obstacles:
  - type:         # str: Type of the obstacle, choose from "Cube," "Sphere," or "Cylinder".
    name:         # str: Name of the object.
    position:     # [float, float, float]: X, Y, Z position.
    orientation:  # [float, float, float, float]: X, Y, Z, W world space quaternion.
    scale:        # [float, float, float]: Scaling along the x, y, z axes.
    color:        # [R, G, B]: Color of the object, range: [0, 1].
    collision:    # bool: True if it is supposed to collide with robots.
    observable:   # bool: True if position and orientation are included in observations for training.
    static:       # bool: False if the object moves in space.
    velocity:     # float: Define the velocity of the trajectory.
    endpoint:     # [float, float, float]: X, Y, Z position towards which the object moves.

    # Extra parameter for objects of type Sphere.
    radius:       # float: Radius of the sphere.

    # Extra parameter for objects of type Cylinder.
    radius:       # float: Radius of the cylinder.
    height:       # float: Height of the cylinder.

```  

### Parameters to define a task/goal
Specific rewards and resets can be defined to create a customized task for agents.

#### Rewards
Rewards are functions that evaluate the current state of the environment, assigning a high value to desirable states and a low value to undesirable ones. The parameters for a reward vary depending on the type. It is important to note that all objects referenced in a reward must exist and be observable.
```yaml  
  rewards:
  - name:                 # str: Name of the reward, freely choosable.
    type:                 # str: Supported types: "Collision," "Distance," "Timestep," or "Shaking".

    # Parameters for type = Collision.
    obj:                  # str: Name of the object to test for collision.
    weight:               # float: Reward/penalty weight.

    # Parameters for type = Distance.
    obj1:                 # str: Name of the object measured from.
    obj2:                 # str: Name of the object measured to.
    distance_weight:      # float: Factor distance is multiplied with.
    orientation_weight:   # float: Factor orientation is multiplied with.
    exponent:             # float: Exponent applied to the final result.
    normalize:            # bool: Normalize reward depending on current position relative to the beginning position.

    # Parameters for type = Timestep.
    weight:               # float: Reward/penalty weight.

    # Parameters for type = Shaking.
    distance:             # str: Name of the distance defined in rewards.
    weight:               # float: Reward/penalty weight.
    length:               # int: Number of past distances taken into account.
``` 

#### Resets
Resets are functions that determine whether the environment needs to be reset. This includes exceeded minimum or maximum values of a previously defined distance function or an exceeded number of timesteps. The specific parameters for a reset differ depending on the type. It is important to note that all objects referenced in a reward must exist and be observable.
```yaml 
  resets:
  - type:         # str: Choose from "CollisionReset," "DistanceReset," "TimestepsReset," or "BoundaryReset".
    reward:       # float: Reward/punishment for a reset.

    # Parameters for type = CollisionReset.
    obj:          # str: Name of the object to test for collision.
    max:          # int: Maximum number of collisions before resetting.

    # Parameters for type = DistanceReset.
    distance:     # str: Name of the distance defined in rewards.
    min_distance: # float: Minimal distance before reset/success.
    max_distance: # float: Maximal distance before resetting.
    max_angle:    # float: Max angular distance between two quaternions before resetting.

    # Parameters for type = TimestepsReset.
    max:          # int: Maximum number of timesteps before resetting.
    min:          # int: Minimal timesteps that have to pass to reset/success.

    # Parameters for type = BoundaryReset.
    obj:          # str: Name of the robot to test for boundary excess (includes the robot's joints).
    max_bound:    # int: Upper boundary.
    min_bound:    # int: Lower boundary.
``` 
