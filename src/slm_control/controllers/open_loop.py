from __future__ import annotations


class OpenLoopController:
    def __init__(self, power: float, maximum_power: float):
        self.power = float(power)
        self.maximum_power = float(maximum_power)

    def reset(self) -> None:
        pass

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del measurement, reference, step_size, index
        return min(max(self.power, 0.0), self.maximum_power)

