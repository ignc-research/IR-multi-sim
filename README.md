# IR-multi-sim
A Framework for robot simulation on multiple engines. It contains the [Isaac Sim](https://developer.nvidia.com/isaac-sim) and the [Pybullet](https://pybullet.org/wordpress/) engine.

## Installation
To use the full application, you need to install IsaacSim aswell as PyBullet.

### Isaac Sim
Follow the instructions found [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) to install Isaac Sim.
Run any python files with Isaacs own python interpreter, usually located at "~/.local/share/ov/package/isaac_sim-*/python.shell", where the * is a wildcard for the installed issac sim version.

### PyBullet
Install the pyb_requirements.txt preferably in a conda environment to use the PyBullet engine.

## Usage
All of MultiSims functionality can be accesssed with configuration files. When running the main.py file, simply specify any of the example configuration files at "./data/configs", or create your own!

```shell
python main.py -f ./data/configs/reach_target_pyb.yaml
```

## Documentation of the spawnable objects and settings
An explanation of the definable settings and how an environment is created can be found [here](docs/configurations.md).

## Full example
This full example demonstrated how to easily create an environment constructed from previously explained parameters.
The example configuration files can be found under "./data/configs".

## Existing models
- [Reach configuration](docs/reach_configuration.md)
- [Set target positions](docs/target_position.md)
