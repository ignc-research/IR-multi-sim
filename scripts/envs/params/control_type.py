from enum import Enum

class ControlType(Enum):
    # Control robots by setting joint velocities
    Velocity = "Velocity"
    # Control robots by directly setting joint positions
    Position = "Position"
    # Control robots by setting their current targets
    PositionTarget = "PositionTarget"