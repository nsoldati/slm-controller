"""Differentiable training for the supported controller families."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from slm_control.config import ExperimentConfig
from slm_control.model import ThermalModel

TRAINABLE_CONTROLLERS = ("open_loop", "output_feedback", "error_feedback", "pi")


@dataclass(frozen=True)
class TrainingResult:
    controller: str
    parameters: dict[str, torch.Tensor]
    losses: torch.Tensor
    stage_losses: dict[str, torch.Tensor] = field(default_factory=dict)


def _add_layer(
    model: ThermalModel, state: torch.Tensor, active_layers: int
) -> tuple[torch.Tensor, int]:
    transition, affine = model.cooling_map(
        active_layers, model.config.time.recoating_duration
    )
    cooled = transition @ state + affine
    powder = torch.full(
        (model.cells,),
        model.config.material.substrate_temperature,
        dtype=model.dtype,
        device=model.device,
    )
    expanded = torch.cat((powder, cooled))
    active_layers += 1
    if active_layers > model.config.roi_layers:
        merged = 0.5 * (
            expanded[-2 * model.cells : -model.cells] + expanded[-model.cells :]
        )
        expanded = torch.cat((expanded[: -2 * model.cells], merged))
        active_layers = model.config.roi_layers
    return expanded, active_layers


def _make_parameters(
    experiment: ExperimentConfig, controller: str, model: ThermalModel
) -> dict[str, torch.nn.Parameter]:
    configured = experiment.controller
    if controller == "open_loop":
        return {
            "power": torch.nn.Parameter(
                torch.tensor(configured.power, dtype=model.dtype, device=model.device)
            )
        }
    if controller in ("output_feedback", "error_feedback"):
        initial_gain = (
            configured.output_gain if controller == "output_feedback" else configured.gain
        )
        return {
            "gains": torch.nn.Parameter(
                torch.full(
                    (experiment.layers, experiment.model.time.steps),
                    initial_gain,
                    dtype=model.dtype,
                    device=model.device,
                )
            )
        }
    if controller == "pi":
        return {
            "proportional_gain": torch.nn.Parameter(
                torch.tensor(configured.gain, dtype=model.dtype, device=model.device)
            ),
            "integral_gain": torch.nn.Parameter(
                torch.tensor(
                    configured.integral_gain, dtype=model.dtype, device=model.device
                )
            ),
        }
    raise ValueError(f"Controller must be one of {TRAINABLE_CONTROLLERS}, got {controller!r}")


def _power(
    experiment: ExperimentConfig,
    controller: str,
    parameters: dict[str, torch.Tensor],
    output: torch.Tensor,
    integral_error: torch.Tensor,
    layer: int,
    index: int,
) -> torch.Tensor:
    configured = experiment.controller
    if controller == "open_loop":
        unconstrained = parameters["power"]
    elif controller == "output_feedback":
        unconstrained = parameters["gains"][layer, index] * output
    elif controller == "error_feedback":
        error = experiment.reference_temperature - output
        unconstrained = configured.feedforward + parameters["gains"][layer, index] * error
    else:
        error = experiment.reference_temperature - output
        unconstrained = (
            configured.feedforward
            + parameters["proportional_gain"] * error
            + parameters["integral_gain"] * integral_error
        )
    return torch.clamp(unconstrained, 0.0, experiment.model.laser.maximum_power)


def _rollout(
    experiment: ExperimentConfig,
    controller: str,
    parameters: dict[str, torch.Tensor],
    model: ThermalModel,
) -> torch.Tensor:
    state = model.initial_state()
    active_layers = 1
    outputs: list[torch.Tensor] = []
    step_size = experiment.model.time.step_size
    steps = experiment.model.time.steps

    for layer in range(experiment.layers):
        if layer > 0:
            state, active_layers = _add_layer(model, state, active_layers)
        system = model.system_matrix(active_layers)
        disturbance = model.disturbance(active_layers)
        integral_error = torch.zeros((), dtype=model.dtype, device=model.device)
        for index in range(steps):
            instant = min(index * step_size, experiment.model.time.printing_duration)
            observation = model.observation_vector(active_layers, experiment.path, instant)
            output = observation @ state
            outputs.append(output)
            if controller == "pi":
                integral_error = integral_error + step_size * (
                    experiment.reference_temperature - output
                )
            power = _power(
                experiment,
                controller,
                parameters,
                output,
                integral_error,
                layer,
                index,
            )
            if index + 1 < steps:
                laser = model.input_vector(active_layers, experiment.path, instant)
                state = state + step_size * (system @ state + laser * power + disturbance)
    return torch.stack(outputs)


def _rollout_first_layer(
    experiment: ExperimentConfig,
    model: ThermalModel,
    gains: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out legacy output feedback and retain its optimized power bias."""
    state = model.initial_state()
    system = model.system_matrix(1)
    disturbance = model.disturbance(1)
    outputs: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    for index in range(experiment.model.time.steps):
        instant = index * experiment.model.time.step_size
        observation = model.observation_vector(1, experiment.path, instant)
        output = observation @ state
        power = torch.clamp(
            gains[index] * output, 0.0, experiment.model.laser.maximum_power
        )
        outputs.append(output)
        powers.append(power)
        if index + 1 < experiment.model.time.steps:
            laser = model.input_vector(1, experiment.path, instant)
            state = state + experiment.model.time.step_size * (
                system @ state + laser * power + disturbance
            )
    return torch.stack(outputs), torch.stack(powers), state


def _train_first_layer_bias(
    experiment: ExperimentConfig,
    model: ThermalModel,
    *,
    epochs: int,
    learning_rate: float,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gains = torch.full(
        (experiment.model.time.steps,),
        experiment.controller.output_gain,
        dtype=model.dtype,
        device=model.device,
        requires_grad=True,
    )
    previous = torch.zeros_like(gains)
    losses: list[torch.Tensor] = []
    for epoch in range(epochs):
        outputs, _, _ = _rollout_first_layer(experiment, model, gains)
        error = outputs - experiment.reference_temperature
        losses.append(torch.mean(error.square()).detach().cpu())
        objective = 0.5 * error.square().sum()
        (gradient,) = torch.autograd.grad(objective, gains)
        with torch.no_grad():
            old = gains.clone()
            if epoch == 0:
                gains -= learning_rate * gradient
            else:
                gains += momentum * (gains - previous) - learning_rate * gradient
                previous.copy_(old)
    gains = gains.detach()
    _, bias_power, final_state = _rollout_first_layer(experiment, model, gains)
    return gains, bias_power.detach(), final_state.detach(), torch.stack(losses)


def _rollout_later_layers(
    experiment: ExperimentConfig,
    model: ThermalModel,
    first_state: torch.Tensor,
    bias_power: torch.Tensor,
    gains: torch.Tensor,
    feedback_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out layers 2..L with one gain schedule shared by every layer."""
    state = first_state
    active_layers = 1
    outputs: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    for _layer in range(1, experiment.layers):
        state, active_layers = _add_layer(model, state, active_layers)
        system = model.system_matrix(active_layers)
        disturbance = model.disturbance(active_layers)
        for index in range(experiment.model.time.steps):
            instant = index * experiment.model.time.step_size
            observation = model.observation_vector(active_layers, experiment.path, instant)
            output = observation @ state
            if index < feedback_start:
                power = bias_power[index]
            else:
                power = bias_power[index] + gains[index] * (
                    output - experiment.reference_temperature
                )
            outputs.append(output)
            powers.append(power)
            if index + 1 < experiment.model.time.steps:
                laser = model.input_vector(active_layers, experiment.path, instant)
                state = state + experiment.model.time.step_size * (
                    system @ state + laser * power + disturbance
                )
    return torch.stack(outputs), torch.stack(powers)


def _later_layer_objective(
    experiment: ExperimentConfig,
    outputs: torch.Tensor,
    powers: torch.Tensor,
    constraint_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = outputs - experiment.reference_temperature
    tracking_loss = torch.mean(error.square())
    maximum = experiment.model.laser.maximum_power
    violation = torch.relu(powers - maximum).square() + torch.relu(-powers).square()
    objective = 0.5 * error.square().sum() + 0.5 * constraint_penalty * violation.sum()
    return tracking_loss, objective


def _acceptable_later_layer_step(
    experiment: ExperimentConfig,
    model: ThermalModel,
    first_state: torch.Tensor,
    bias: torch.Tensor,
    candidate: torch.Tensor,
    feedback_start: int,
    constraint_penalty: float,
    current_objective: torch.Tensor,
) -> bool:
    """Check one proposed update without building another autograd graph."""
    with torch.no_grad():
        outputs, powers = _rollout_later_layers(
            experiment, model, first_state, bias, candidate, feedback_start
        )
        _, candidate_objective = _later_layer_objective(
            experiment, outputs, powers, constraint_penalty
        )
    return bool(
        torch.isfinite(candidate_objective) and candidate_objective <= current_objective
    )


def train_legacy_multilayer_controller(
    experiment: ExperimentConfig,
    *,
    epochs: int = 75,
    learning_rate: float = 3e-6,
    first_layer_epochs: int = 120,
    first_layer_learning_rate: float = 8e-9,
    feedback_start: int = 10,
    constraint_penalty: float = 100.0,
) -> TrainingResult:
    """Train the legacy two-stage multilayer bias/error-feedback algorithm."""
    if experiment.layers < 2:
        raise ValueError("legacy multilayer training requires at least two layers")
    if epochs < 1 or first_layer_epochs < 1:
        raise ValueError("training epochs must be positive")
    if learning_rate <= 0 or first_layer_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if feedback_start < 0:
        raise ValueError("feedback_start cannot be negative")
    if constraint_penalty < 0:
        raise ValueError("constraint_penalty cannot be negative")
    model = ThermalModel(experiment.model)
    single_layer = replace(experiment, layers=1)
    first_gains, bias, first_state, first_losses = _train_first_layer_bias(
        single_layer,
        model,
        epochs=first_layer_epochs,
        learning_rate=first_layer_learning_rate,
        momentum=0.5,
    )
    gains = torch.full(
        (experiment.model.time.steps,),
        experiment.controller.gain,
        dtype=model.dtype,
        device=model.device,
        requires_grad=True,
    )
    velocity = torch.zeros_like(gains)
    current_learning_rate = learning_rate
    momentum = 0.5
    losses: list[torch.Tensor] = []
    for epoch in range(epochs):
        outputs, powers = _rollout_later_layers(
            experiment, model, first_state, bias, gains, feedback_start
        )
        tracking_loss, objective = _later_layer_objective(
            experiment, outputs, powers, constraint_penalty
        )
        if not torch.isfinite(objective):
            raise RuntimeError(f"stage-two objective became non-finite at epoch {epoch + 1}")
        losses.append(tracking_loss.detach().cpu())
        # This objective has the same terms as the legacy manual Jacobian
        # implementation. Power remains unclipped while training.
        objective = objective + 0.0 * gains.sum()
        (gradient,) = torch.autograd.grad(objective, gains)
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"stage-two gradient became non-finite at epoch {epoch + 1}")
        if torch.mean(gradient) < 0:
            momentum = 0.0
        proposed_velocity = momentum * velocity - current_learning_rate * gradient
        candidate = (gains + proposed_velocity).detach()
        accepted = _acceptable_later_layer_step(
            experiment,
            model,
            first_state,
            bias,
            candidate,
            feedback_start,
            constraint_penalty,
            objective.detach(),
        )
        if accepted:
            with torch.no_grad():
                gains.copy_(candidate)
                velocity = proposed_velocity
        else:
            momentum = 0.0
            velocity.zero_()
            current_learning_rate *= 0.5
        if epoch > 0:
            current_learning_rate *= 0.99
    return TrainingResult(
        controller="error_feedback",
        parameters={
            "first_layer_gains": first_gains.cpu(),
            "bias_power": bias.cpu(),
            "later_layer_gains": gains.detach().cpu(),
            "feedback_start": torch.tensor(feedback_start),
        },
        losses=torch.stack(losses),
        stage_losses={"first_layer": first_losses, "later_layers": torch.stack(losses)},
    )


def train_controller(
    experiment: ExperimentConfig,
    controller: str,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-6,
    first_layer_epochs: int = 120,
    first_layer_learning_rate: float = 8e-9,
    feedback_start: int = 10,
    constraint_penalty: float = 100.0,
) -> TrainingResult:
    """Train a controller on either a single- or multi-layer configuration."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if controller == "error_feedback" and experiment.layers > 1:
        return train_legacy_multilayer_controller(
            experiment,
            epochs=epochs,
            learning_rate=learning_rate,
            first_layer_epochs=first_layer_epochs,
            first_layer_learning_rate=first_layer_learning_rate,
            feedback_start=feedback_start,
            constraint_penalty=constraint_penalty,
        )
    model = ThermalModel(experiment.model)
    parameters = _make_parameters(experiment, controller, model)
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    losses: list[torch.Tensor] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        output = _rollout(experiment, controller, parameters, model)
        loss = torch.mean((output - experiment.reference_temperature) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu())
    return TrainingResult(
        controller=controller,
        parameters={name: value.detach().cpu() for name, value in parameters.items()},
        losses=torch.stack(losses),
    )


def train_single_layer_gains(
    experiment: ExperimentConfig,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-6,
) -> TrainingResult:
    """Compatibility wrapper for legacy callers of the original trainer."""
    if experiment.layers != 1:
        raise ValueError("This wrapper requires a single-layer experiment")
    return train_controller(
        experiment, "error_feedback", epochs=epochs, learning_rate=learning_rate
    )
