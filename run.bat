REM If this is not working, use conda init in the consol and restart terminal

REM Define the python path to the isaac interpreter here
set PYTHON_PATH=C:\Users\chris\AppData\Local\ov\pkg\isaac_sim-2022.2.1\python.bat

REM Define anaconda environment name here
set CONDA_ENV_NAME=ir-multi-sim

REM Check if the first argument is "pyb"
if "%1" equ "pybullet" (
    REM Check if defined conda environment exists
    conda info --envs | findstr /c:"%CONDA_ENV_NAME%" >nul
    if errorlevel 1 (
        echo Error: Conda environment "%CONDA_ENV_NAME%" not found.
        exit /b 1
    )

    rem Activate the conda environment
    call activate %CONDA_ENV_NAME%

    rem Run the python script in that environment
    python main.py %1 %2 %3 %4 

    rem Deactivate the conda environment
    conda deactivate

) else if "%1" equ "isaac" (
    REM Check if defined python path exists
    if not exist "%PYTHON_PATH%" (
        echo Error: Invalid Python path: %PYTHON_PATH%
        exit /b 1
    )

    REM Run python code with the chosen interpreter and all arguments except the first one
    %PYTHON_PATH% main.py %1 %2 %3 %4

) else (
    REM If no valid Engine is given, print an error message
    echo Error: You need to specify a valid engine!
    exit /b 1
)
