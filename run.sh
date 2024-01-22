#!/bin/bash

# If this is not working, use conda init in the console and restart terminal

# Define the python path to the isaac interpreter here
PYTHON_PATH="/mnt/c/Users/chris/AppData/Local/ov/pkg/isaac_sim-2022.2.1/kit/exts/omni.kit.window.extensions/ext_template/tools/python.sh"

# Define anaconda environment name here
CONDA_ENV_NAME="ir-multi-sim"

# Check if the first argument is "pybullet"
if [ "$1" == "pybullet" ]; then
    # Check if the defined conda environment exists
    if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
        echo "Error: Conda environment '$CONDA_ENV_NAME' not found."
        exit 1
    fi

    # Activate the conda environment
    source activate "$CONDA_ENV_NAME"

    # Run the python script in that environment
    python main.py $1 $2 $3

    # Deactivate the conda environment
    conda deactivate

elif [ "$1" == "isaac" ]; then
    # Check if the defined python path exists
    if [ ! -x "$PYTHON_PATH" ]; then
        echo "Error: Invalid Python path: $PYTHON_PATH"
        exit 1
    fi

    # Run Python code with the chosen interpreter and all arguments except the first one
    "$PYTHON_PATH" main.py $1 $2 $3

else
    # If no valid engine is given, print an error message
    echo "Error: You need to specify a valid engine!"
    exit 1
fi
