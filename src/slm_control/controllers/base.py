"""Common controller protocol and factory."""

from __future__ import annotations

from typing import Protocol

from slm_control.config import ControllerConfig


class Controller(Protocol):
    def reset(self) -> None: ...

    def compute(
        self, measurement: float, reference: float, step_size: float, index: int
    ) -> float: ...


def build_controller(config: ControllerConfig, maximum_power: float) -> Controller:
    from slm_control.controllers.feedback import (
        ErrorFeedbackController,
        OutputFeedbackController,
        PIController,
    )
    from slm_control.controllers.open_loop import OpenLoopController

    kind = config.kind.lower()
    if kind == "open_loop":
        return OpenLoopController(config.power, maximum_power)
    if kind == "output_feedback":
        return OutputFeedbackController(config.output_gain, maximum_power)
    if kind == "error_feedback":
        return ErrorFeedbackController(config.feedforward, config.gain, maximum_power)
    if kind == "pi":
        return PIController(
            config.feedforward,
            config.gain,
            config.integral_gain,
            maximum_power,
        )
    raise ValueError(f"Unknown controller kind {config.kind!r}")
