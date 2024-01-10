# Documentation of the spawnable objects and usable settings
The yaml settings file consists of multiple definable elements focusing on the runtime settings itself as well as the specific environment. 

## General runtime parameter 
```yaml
run:
  engine:                   # str: "PyBullet" or "Isaac"
  load_model:               # bool: create new model or use an existing one 
  load_name:                # str: need to be a string of the path to the .zip file (starting with the model name, not use \\ for \) 
  save_name:                # str: name of the original model you want to continue training
  algorithm:                # possibility to define custom algorithm settings. Otherwise a default is used
    type:                   # str: "PPO", "TD3", "SAC", "A2C" or "DDPG"
    parameters:             # define variety of parameters for your model e.g.:
      gamma:                # float: Reinforcement learning specific parameter
      learning_rate:        # float: learing rate of the model
      batch_size:           # int: batch size for the model
      ...
    custom_policy:          # define specific layer settings and activation function
      activation_function:    # str: "RELU" or "Tanh"
      value_function:
        - 256
        - 256
        - 256
        - 256
      policy_function:
        - 256
        - 256
        - 256
        - 256
```

## Training and evaluation parameter
```yaml 
train:  	
  logging:      # int: level of information to be output during learing 0, 1, 2, ...
  timesteps:    # int: amount of timesteps the model should learn
  save_freq:    # int: how frequently a model should be saved


evaluation:
  timesteps:    # int: amount of timesteps the model ueses to evaluate
```

## Environment parameter
```yaml 
env:
  num_envs:     # int: number of simultaneously running environments
  env_offset:   # [int, int]: size in x,y direction of an environment
  headless:     # bool: run with visualization
  step_size:    # float: size of one simulation step, in general 240Hz = 1/240
  step_count:   # int: amount of steps the simulation can do each iteration
```

### Objects within an environment
You can create as many object of a type as needed. To do so, start a new object definition under the specific object type with a new hyphen. Instead of specific sensors, this project works with general observability definable for all objects. Therefore, robots, robot joints and obstacles can be marked as observable, meaning that their relative position, orientation and scale will be included in the observations of the machine learning model.

#### Robots
To create a robot in an environment, the urdf path is the only required argument. All other parameters are optional. The robots need to be defined in an [urdf](http://wiki.ros.org/urdf) file which needs to be saved in the "./data/robots" directory.
```yaml 
  robots:
    - name:                 # str: name of the robot  
      urdf_path:            # str: path to the urdf, usually in the robots folder
      position:             # [float, float, float]: x,y,z base position of the robot
      orientation:          # [float, float, float]: x,y,z base position of the robot
      collision:            # bool: true if robot is supposed to collide with surroundings
      observable:           # bool: true if pos and orientation included in observations for training
      observable_joints:    # ["string", ...]: robot joint names that should be observerd
      control_type:         # str: "Velocity" or "Position" to define control type
      max_velocity:         # float: define maximal velocity a joint can be moved by 
    
    - name:                 # start new robot object
```

#### General URDFs
To define a general urdf object, you need to provide a path to an urdf file. The other parameters are optional. 
```yaml 
  urdfs:
    - name:         # str: name of the urdf
      urdf_path:    # str: path of the urdf
      scale:        # [float, float, float]: along x-, y-, z-axis
      position:     # [float, float, float]: x,y,z position
      orientation:  # [float, float, float]: x,y,z orientation
```  

#### Spawnable Objects
To define spawnable objects, you need to provide the object typ. The other parameters are optional. You can define the position, orientation and scale for each object a deterministic with a 3-dimensional list. Additionally, you can define ranges with minimal and maximal valid values (e.g.: [[min, min, min], [max, max, max]]) to automatically create randomized values for each reset.
```yaml 
  obstacles:
    - type:         # str: "Cube", "Spher" or "Cylinder"
      name:         # str: name of the object
      position:     # [float, float, float]: x,y,z position
      orientation:  # [float, float, float]: x,y,z orientation
      scale:        # [floa, float, float]: scale along x-, y-, z-axis 
      color:        # [R,G,B]: color of the robot, range: [0,1]
      collision:    # bool: true if robot is supposed to collide with surroundings
      observable:   # bool: true if pos and orientation included in observations for training
      static:       # bool: false if the object moves in space 
      velocity:     # float: define the velocity of the tracjectory  
      endpoint:     # [float, float, float]: x,y,z position towards an object moves

      # extra parameter for object of type sphere
      radius:       # float: define the velocity of the tracjectory  

      # extra parameter for object of type cylinder
      radius:       # float: define the velocity of the tracjectory  
      height:       # float: define the velocity of the tracjectory  
```  

### Parameters to define a task/goal
You can define specific rewards and resets to create a custom task for your agents.

#### Rewards
Rewards are functions which evluate the current environment state, rating desirable states with a high value and undesirable states with a low value. The specific parameters for a reward differ depending on the type. Also note that all objects referenced in a rewards must exist and have to be observable
```yaml 
  rewards:
    - name:                 # str: name of the reward, free choosable 
      type:                 # str:  supported types: "collision", "distance" or "timestep"

      # parameters for type = collision 
      obj:                  # str: name of the object you want to test for collision 
      weight:               # float: reward/penalty

      # parameters for type = distance  
      obj1:                 # str: name of object measured form   
      obj2:                 # str: name of object measured to
      distance_weight:      # float: Factor distance is multiplied with
      orientation_weight:   # float: Factor orientation is multiplied with
      exponent:             # float: Exponent applied to final result
      normalize:            # bool: normalize reward depending on current pos relative to beginning pos
    
      # parameters for type = timestep
      weight: # float: reward/penalty
``` 

#### Resets
Resets are functions which determine wether the environment needs to be reset. This included exceeded min or max values of a previously defined distance function, or an exceeded number of timesteps. The specific parameters for a reset differ depending on the type. Also note that all objects referenced in a rewards must exist and have to be observable
```yaml 
  resets:  
    - type:         # str: "CollisionReset", "DistanceReset" or "TimestepsReset"
      reward:       # float: reward/ punishment for a reset

      # parameters for type = CollisionReset 
      obj:          # str: name of the object you want to test for collision 
      max:          # int: max amount of collision before reseting

      # parameters for type = DistanceReset 
      distance:     # str: name of the distance defined in rewards 
      min_distance: # float: minimal distance before reset/success
      max_distance: # float: maximal distance before resetting
      max_angle:    # float: max angular distance between two quaternions before resetting 
      
      # parameters for type = TimestepsReset 
      max:          # int: max amount of timesteps before resetting
      min:          # int: minimal timesteps that have to pass to reset/success
``` 
