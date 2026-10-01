"""Console entry points for reproducible simulations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from slm_control.config import ModelConfig, load_experiment
from slm_control.controllers import (
    GainScheduleController,
    LayerwiseBiasFeedbackController,
    OpenLoopController,
    OutputGainScheduleController,
    PIController,
)
from slm_control.io import save_run
from slm_control.simulation import simulate_multilayer
from slm_control.training import TRAINABLE_CONTROLLERS, train_controller
from slm_control.trajectories import available_paths
from slm_control.visualization import plot_tracking, plot_trajectory


def _run(arguments: Sequence[str] | None, expected_layers: int | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an SLM thermal-control experiment")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate a tracking plot")
    options = parser.parse_args(arguments)
    experiment = load_experiment(options.config)
    if expected_layers == 1 and experiment.layers != 1:
        parser.error("the single-layer command requires layers: 1")
    result = simulate_multilayer(experiment)
    run_directory = save_run(experiment, result)
    if not options.no_plot:
        figure = Path(experiment.figure_root) / "evaluations" / f"{run_directory.name}.png"
        plot_tracking(result, figure)
        print(f"Figure saved to {figure}")
    print(f"Run saved to {run_directory}")
    print(f"RMSE: {result.rmse:.6g} K")
    return 0


def single_layer(arguments: Sequence[str] | None = None) -> int:
    return _run(arguments, expected_layers=1)


def multilayer(arguments: Sequence[str] | None = None) -> int:
    return _run(arguments)


def plot_trajectory_command(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot an SLM laser trajectory")
    parser.add_argument("--figure", required=True, choices=available_paths())
    parser.add_argument(
        "--config", help="Optional experiment YAML providing geometry and duration"
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    options = parser.parse_args(arguments)
    if options.config:
        experiment = load_experiment(options.config)
        model = experiment.model
        figure_root = Path(experiment.figure_root)
    else:
        model = ModelConfig()
        figure_root = Path("figures")
    output = options.output or figure_root / "trajectories" / f"{options.figure}.png"
    plot_trajectory(
        options.figure,
        model,
        output,
        samples=options.samples,
        show=options.show,
    )
    print(f"Trajectory saved to {output}")
    return 0


def _apply_scope(experiment, scope: str, parser: argparse.ArgumentParser):
    if scope == "single":
        return replace(experiment, layers=1)
    if experiment.layers < 2:
        parser.error("--layers multi requires layers >= 2 in the experiment YAML")
    return experiment


def train_controller_command(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an SLM controller")
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", required=True, choices=("single", "multi"))
    parser.add_argument("--controller", required=True, choices=TRAINABLE_CONTROLLERS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--first-layer-epochs",
        type=int,
        default=120,
        help="Output-feedback epochs in stage one of multi error-feedback training",
    )
    parser.add_argument(
        "--first-layer-learning-rate",
        type=float,
        default=8e-9,
        help="Stage-one learning rate (legacy square-spiral default: 8e-9)",
    )
    parser.add_argument(
        "--feedback-start",
        type=int,
        default=10,
        help="First sample using feedback on layers after the first",
    )
    parser.add_argument(
        "--constraint-penalty",
        type=float,
        default=100.0,
        help="Soft power-constraint penalty used during multilayer training",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    experiment = _apply_scope(load_experiment(options.config), options.layers, parser)
    experiment = replace(
        experiment, controller=replace(experiment.controller, kind=options.controller)
    )
    result = train_controller(
        experiment,
        options.controller,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        first_layer_epochs=options.first_layer_epochs,
        first_layer_learning_rate=options.first_layer_learning_rate,
        feedback_start=options.feedback_start,
        constraint_penalty=options.constraint_penalty,
    )
    tensors = [result.losses, *result.parameters.values(), *result.stage_losses.values()]
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("training produced non-finite values; checkpoint was not saved")
    output = options.output or Path(
        f"artifacts/checkpoints/{options.controller}_{options.layers}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "controller": result.controller,
            "scope": options.layers,
            "layers": experiment.layers,
            "steps_per_layer": experiment.model.time.steps,
            "epochs": options.epochs,
            "learning_rate": options.learning_rate,
            "first_layer_epochs": options.first_layer_epochs,
            "first_layer_learning_rate": options.first_layer_learning_rate,
            "feedback_start": options.feedback_start,
            "constraint_penalty": options.constraint_penalty,
            "parameters": result.parameters,
            "losses": result.losses,
            "stage_losses": result.stage_losses,
            "experiment": experiment.to_dict(),
        },
        output,
    )
    print(f"Checkpoint saved to {output}")
    if "first_layer" in result.stage_losses:
        first_losses = result.stage_losses["first_layer"]
        print(f"First-layer initial loss: {float(first_losses[0]):.6g}")
        print(f"First-layer final loss: {float(first_losses[-1]):.6g}")
        print(f"Later-layers initial loss: {float(result.losses[0]):.6g}")
        print(f"Later-layers final loss: {float(result.losses[-1]):.6g}")
    else:
        print(f"Initial loss: {float(result.losses[0]):.6g}")
        print(f"Final loss: {float(result.losses[-1]):.6g}")
    return 0


def _controller_from_checkpoint(checkpoint: dict, experiment):
    kind = checkpoint["controller"]
    parameters = checkpoint["parameters"]
    maximum = experiment.model.laser.maximum_power
    configured = experiment.controller
    if kind == "open_loop":
        return OpenLoopController(float(parameters["power"]), maximum)
    if kind == "output_feedback":
        return OutputGainScheduleController(parameters["gains"], maximum)
    if kind == "error_feedback":
        if "bias_power" in parameters:
            return LayerwiseBiasFeedbackController(
                parameters["first_layer_gains"],
                parameters["bias_power"],
                parameters["later_layer_gains"],
                maximum,
                int(parameters["feedback_start"]),
            )
        return GainScheduleController(configured.feedforward, parameters["gains"], maximum)
    if kind == "pi":
        return PIController(
            configured.feedforward,
            float(parameters["proportional_gain"]),
            float(parameters["integral_gain"]),
            maximum,
        )
    raise ValueError(f"Unsupported checkpoint controller {kind!r}")


def evaluate_controller_command(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained SLM controller")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", required=True, choices=("single", "multi"))
    parser.add_argument("--controller", required=True, choices=TRAINABLE_CONTROLLERS)
    parser.add_argument("--no-plot", action="store_true")
    options = parser.parse_args(arguments)
    experiment = _apply_scope(load_experiment(options.config), options.layers, parser)
    experiment = replace(
        experiment, controller=replace(experiment.controller, kind=options.controller)
    )
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("controller") != options.controller:
        parser.error(
            f"checkpoint contains {checkpoint.get('controller')!r}, not {options.controller!r}"
        )
    if checkpoint.get("scope") != options.layers:
        parser.error(
            f"checkpoint was trained for {checkpoint.get('scope')!r}, not {options.layers!r}"
        )
    if checkpoint.get("layers") != experiment.layers:
        parser.error("checkpoint and evaluation configuration use different layer counts")
    if checkpoint.get("steps_per_layer") != experiment.model.time.steps:
        parser.error("checkpoint and evaluation configuration use different time grids")
    controller = _controller_from_checkpoint(checkpoint, experiment)
    result = simulate_multilayer(experiment, controller)
    run_directory = save_run(experiment, result)
    if not options.no_plot:
        figure = Path(experiment.figure_root) / "evaluations" / f"{run_directory.name}.png"
        plot_tracking(result, figure)
        print(f"Figure saved to {figure}")
    print(f"Evaluation saved to {run_directory}")
    print(f"RMSE: {result.rmse:.6g} K")
    return 0
