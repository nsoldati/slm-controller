"""Experiment-independent single- and multi-layer simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slm_control.config import ExperimentConfig
from slm_control.controllers import Controller, build_controller
from slm_control.model import ThermalModel


@dataclass(frozen=True)
class SimulationResult:
    time: torch.Tensor
    output: torch.Tensor
    power: torch.Tensor
    reference: torch.Tensor
    state: torch.Tensor
    layer: torch.Tensor

    @property
    def rmse(self) -> float:
        return float(torch.sqrt(torch.mean((self.output - self.reference) ** 2)))


def _cool(model: ThermalModel, state: torch.Tensor, layers: int) -> torch.Tensor:
    duration = model.config.time.recoating_duration
    if duration == 0:
        return state
    transition, affine = model.cooling_map(layers, duration)
    return transition @ state + affine


def _add_powder_layer(
    model: ThermalModel, state: torch.Tensor, layers: int
) -> tuple[torch.Tensor, int]:
    cooled = _cool(model, state, layers)
    new_layer = torch.full(
        (model.cells,),
        model.config.material.substrate_temperature,
        dtype=model.dtype,
        device=model.device,
    )
    expanded = torch.cat((new_layer, cooled))
    new_count = layers + 1
    maximum = model.config.roi_layers
    if new_count > maximum:
        # Preserve a fixed ROI by merging the two oldest layer fields.
        merged_bottom = 0.5 * (expanded[-2 * model.cells : -model.cells] + expanded[-model.cells :])
        expanded = torch.cat((expanded[: -2 * model.cells], merged_bottom))
        new_count = maximum
    return expanded, new_count


def _simulate_layer(
    model: ThermalModel,
    state: torch.Tensor,
    layers: int,
    path: str,
    reference: float,
    controller: Controller,
    noise_std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = model.config.time
    times = torch.linspace(
        0.0,
        config.printing_duration,
        config.steps,
        dtype=model.dtype,
        device=model.device,
    )
    states = torch.empty((config.steps, state.numel()), dtype=model.dtype, device=model.device)
    outputs = torch.empty(config.steps, dtype=model.dtype, device=model.device)
    powers = torch.empty(config.steps, dtype=model.dtype, device=model.device)
    states[0] = state
    matrix = model.system_matrix(layers)
    disturbance = model.disturbance(layers)
    controller.reset()

    for index, instant in enumerate(times):
        observation = model.observation_vector(layers, path, float(instant))
        measurement = observation @ states[index]
        if noise_std:
            measurement = measurement + noise_std * torch.randn(
                (), dtype=model.dtype, device=model.device, generator=generator
            )
        outputs[index] = measurement
        powers[index] = controller.compute(float(measurement), reference, config.step_size, index)
        if index + 1 < config.steps:
            input_vector = model.input_vector(layers, path, float(instant))
            derivative = matrix @ states[index] + input_vector * powers[index] + disturbance
            step = float(times[index + 1] - instant)
            states[index + 1] = states[index] + step * derivative
    return times, states, outputs, powers


def simulate_multilayer(
    experiment: ExperimentConfig,
    controller: Controller | None = None,
) -> SimulationResult:
    """Run a configured experiment, reducing old layers to the configured ROI."""
    model = ThermalModel(experiment.model)
    active_layers = 1
    state = model.initial_state()
    controller = controller or build_controller(
        experiment.controller, experiment.model.laser.maximum_power
    )
    generator = torch.Generator(device=model.device).manual_seed(experiment.seed)
    time_parts: list[torch.Tensor] = []
    state_parts: list[torch.Tensor] = []
    output_parts: list[torch.Tensor] = []
    power_parts: list[torch.Tensor] = []
    layer_parts: list[torch.Tensor] = []
    max_states = min(experiment.layers, experiment.model.roi_layers) * model.cells

    for layer_number in range(1, experiment.layers + 1):
        if layer_number > 1:
            state, active_layers = _add_powder_layer(model, state, active_layers)
        local_time, local_states, outputs, powers = _simulate_layer(
            model,
            state,
            active_layers,
            experiment.path,
            experiment.reference_temperature,
            controller,
            experiment.measurement_noise_std,
            generator,
        )
        state = local_states[-1]
        offset = (layer_number - 1) * experiment.model.time.printing_duration
        time_parts.append(local_time + offset)
        padded = torch.full(
            (len(local_time), max_states),
            float("nan"),
            dtype=model.dtype,
            device=model.device,
        )
        padded[:, : local_states.shape[1]] = local_states
        state_parts.append(padded)
        output_parts.append(outputs)
        power_parts.append(powers)
        layer_parts.append(
            torch.full((len(local_time),), layer_number, dtype=torch.int64, device=model.device)
        )

    output = torch.cat(output_parts)
    return SimulationResult(
        time=torch.cat(time_parts),
        output=output,
        power=torch.cat(power_parts),
        reference=torch.full_like(output, experiment.reference_temperature),
        state=torch.cat(state_parts),
        layer=torch.cat(layer_parts),
    )
