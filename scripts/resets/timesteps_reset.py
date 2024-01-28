from scripts.resets.reset import Reset

_timeReset = 0

class TimestepsReset(Reset):
    def __init__(self, 
                 max: int, 
                 min: int=None, 
                 reward: float=0, 
                 name: str=None
                ) -> None:
        
        super().__init__(reward)

        self.max = max
        self.min = min
        self.name = name if name else "TimeReset_" + f"{_timeReset+1}"

        