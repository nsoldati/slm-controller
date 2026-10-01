"""Output-feedback controllers with explicit error convention."""

from __future__ import annotations

import torch


def _clip(value: float, upper: float) -> float:
    return min(max(value, 0.0), upper)


class OutputFeedbackController:
    """Legacy output feedback using power = gain * measurement."""

    def __init__(self, gain: float, maximum_power: float):
        self.gain = float(gain)
        self.maximum_power = float(maximum_power)

    def reset(self) -> None:
        pass

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del reference, step_size, index
        return _clip(self.gain * measurement, self.maximum_power)


class OutputGainScheduleController:
    """Legacy output feedback with one trained gain per layer and sample."""

    def __init__(self, gains: torch.Tensor, maximum_power: float):
        if gains.ndim not in (1, 2):
            raise ValueError("gains must have shape (time,) or (layer, time)")
        self.gains = gains.detach().cpu()
        self.maximum_power = float(maximum_power)
        self.layer = -1

    def reset(self) -> None:
        self.layer += 1

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del reference, step_size
        gains = self.gains if self.gains.ndim == 1 else self.gains[self.layer]
        if index >= len(gains):
            raise IndexError("gain schedule is shorter than the simulation")
        return _clip(float(gains[index]) * measurement, self.maximum_power)


class LayerwiseBiasFeedbackController:
    """Legacy two-stage controller for a complete multilayer build.

    Layer one uses output feedback. Every later layer reuses the optimized
    first-layer power as a time-varying bias and the same error-gain schedule.
    """

    def __init__(
        self,
        first_layer_gains: torch.Tensor,
        bias_power: torch.Tensor,
        later_layer_gains: torch.Tensor,
        maximum_power: float,
        feedback_start: int = 10,
    ):
        schedules = (first_layer_gains, bias_power, later_layer_gains)
        if any(schedule.ndim != 1 for schedule in schedules):
            raise ValueError("multilayer schedules must be one-dimensional")
        if len({len(schedule) for schedule in schedules}) != 1:
            raise ValueError("multilayer schedules must have equal lengths")
        self.first_layer_gains = first_layer_gains.detach().cpu()
        self.bias_power = bias_power.detach().cpu()
        self.later_layer_gains = later_layer_gains.detach().cpu()
        self.maximum_power = float(maximum_power)
        self.feedback_start = int(feedback_start)
        self.layer = -1

    def reset(self) -> None:
        self.layer += 1

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del step_size
        if index >= len(self.bias_power):
            raise IndexError("controller schedule is shorter than the simulation")
        if self.layer == 0:
            power = float(self.first_layer_gains[index]) * measurement
        elif index < self.feedback_start:
            power = float(self.bias_power[index])
        else:
            # The sign intentionally matches the legacy implementation.
            error = measurement - reference
            power = float(self.bias_power[index]) + float(self.later_layer_gains[index]) * error
        return _clip(power, self.maximum_power)


class ErrorFeedbackController:
    """Output feedback using error = reference - measurement."""

    def __init__(self, feedforward: float, gain: float, maximum_power: float):
        self.feedforward = float(feedforward)
        self.gain = float(gain)
        self.maximum_power = float(maximum_power)

    def reset(self) -> None:
        pass

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del step_size, index
        error = reference - measurement
        return _clip(self.feedforward + self.gain * error, self.maximum_power)


class GainScheduleController:
    """Error-feedback controller with one trained gain per time sample."""

    def __init__(self, feedforward: float, gains: torch.Tensor, maximum_power: float):
        if gains.ndim not in (1, 2):
            raise ValueError("gains must have shape (time,) or (layer, time)")
        self.feedforward = float(feedforward)
        self.gains = gains.detach().cpu()
        self.maximum_power = float(maximum_power)
        self.layer = -1

    def reset(self) -> None:
        self.layer += 1

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del step_size
        gains = self.gains if self.gains.ndim == 1 else self.gains[self.layer]
        if index >= len(gains):
            raise IndexError("gain schedule is shorter than the simulation")
        error = reference - measurement
        return _clip(self.feedforward + float(gains[index]) * error, self.maximum_power)


class PIController:
    """PI controller using the conventional error = reference - measurement."""

    def __init__(
        self,
        feedforward: float,
        proportional_gain: float,
        integral_gain: float,
        maximum_power: float,
    ):
        self.feedforward = float(feedforward)
        self.proportional_gain = float(proportional_gain)
        self.integral_gain = float(integral_gain)
        self.maximum_power = float(maximum_power)
        self.integral_error = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def compute(self, measurement: float, reference: float, step_size: float, index: int) -> float:
        del index
        error = reference - measurement
        self.integral_error += step_size * error
        unconstrained = (
            self.feedforward
            + self.proportional_gain * error
            + self.integral_gain * self.integral_error
        )
        return _clip(unconstrained, self.maximum_power)
