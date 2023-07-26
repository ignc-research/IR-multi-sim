from enum import Enum

class ControlType(Enum):
    # Control robots by setting joint velocities
    VELOCITY = 1,
    # Control robots by directly setting joint positions
    POSITION = 2