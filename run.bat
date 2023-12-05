REM If this is not working, use conda init in the consol and restart terminal

REM Check if the first argument is "pyb"
if "%1" equ "pyb" (
    rem Activate the conda environment
    call activate ir-multi-sim
    rem Run the python script in that environment
    python main.py %2 %3 %4 %5 %6 %7 %8 %9
    rem Deactivate the conda environment
    conda deactivate

) else if "%1" equ "isaac" (
    REM Run your Python code with the chosen interpreter and all arguments except the first one
    C:\Users\chris\AppData\Local\ov\pkg\isaac_sim-2022.2.1\python.bat main.py %2 %3 %4 %5 %6 %7 %8 %9

) else (
    REM If no valid Engine is given, print an error message
    echo Error: You need to specify an Engine.
    exit /b 1
)
