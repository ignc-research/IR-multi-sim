from scripts.resets.reset import Reset

class TimestepsReset(Reset):
    def __init__(self, max: int) -> None:
        self.max = max