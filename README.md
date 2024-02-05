# IR-multi-sim
A Framework for robot simulation on multiple engines. It contains the [Isaac Sim](https://developer.nvidia.com/isaac-sim) and the [Pybullet](https://pybullet.org/wordpress/) engine.

## Installation
To use the full application, you need to install Isaac Sim aswell as PyBullet.

### Isaac Sim
Follow the instructions found [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) to install Isaac Sim.
Run any python files with Isaacs own python interpreter, usually located at "~/.local/share/ov/package/isaac_sim-*/python.shell", where the * is a wildcard for the installed issac sim version.

### PyBullet
Install the requirements.txt in a conda environment to use the PyBullet engine.

## Usage
The multi-sim can be started via the main.py file directly or through the run.bat for Windows and run.sh for Ubuntu. Include a parameter for the desired engine and a path to the configuration file, usually located in the configs directory. For further information, usethe -h flag for help. Here are some example calls:

```shell
python main.py pybullet ./configs/simple_test_pyb.yaml
```

 Or run with the shell script that handles switching between interpreters:

```shell
.\run.bat isaac ./configs/simple_test_isaac.yaml
```

## Documentation of the spawnable objects and settings
An explanation of the definable settings and how an environment is created can be found [here](docs/configurations.md).

## Full example
This full example demonstrated how to easily create an environment constructed from previously explained parameters.
The example configuration files can be found under "./configs".

## Existing models
A simple model is available at '/models/' that implements the task of setting the target position with both simulators. For further details, refer to [this link](docs/target_position.md).