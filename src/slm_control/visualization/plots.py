"""Publication-friendly plots driven by simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slm_control.config import ModelConfig
from slm_control.simulation import SimulationResult
from slm_control.trajectories import position


def plot_tracking(result: SimulationResult, destination: str | Path | None = None) -> None:
    time_ms = result.time.detach().cpu().numpy() * 1e3
    figure, temperature_axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    temperature_axis.plot(
        time_ms, result.reference.cpu(), "--", color="darkorange", label="Reference"
    )
    temperature_axis.plot(time_ms, result.output.detach().cpu(), color="dodgerblue", label="Output")
    temperature_axis.set(xlabel="Time [ms]", ylabel="Output [K]")
    temperature_axis.grid(alpha=0.2)
    power_axis = temperature_axis.twinx()
    power_axis.plot(
        time_ms,
        result.power.detach().cpu(),
        color="seagreen",
        alpha=0.75,
        label="Power",
    )
    power_axis.set_ylabel("Laser power [W]")
    lines = temperature_axis.lines + power_axis.lines
    temperature_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    if destination is None:
        plt.show()
    else:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200)
        plt.close(figure)


def plot_trajectory(
    path: str,
    model: ModelConfig,
    destination: str | Path | None = None,
    *,
    samples: int = 500,
    show: bool = False,
) -> None:
    """Plot a configured laser trajectory in micrometres."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    duration = model.time.printing_duration
    times = np.linspace(0.0, duration, samples)
    coordinates = np.asarray(
        [position(path, instant, duration, model.geometry) for instant in times]
    )
    figure, axis = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    axis.plot(coordinates[:, 0] * 1e6, coordinates[:, 1] * 1e6, linewidth=3)
    axis.scatter(
        coordinates[0, 0] * 1e6,
        coordinates[0, 1] * 1e6,
        color="seagreen",
        label="Start",
        zorder=3,
    )
    axis.scatter(
        coordinates[-1, 0] * 1e6,
        coordinates[-1, 1] * 1e6,
        color="firebrick",
        label="End",
        zorder=3,
    )
    axis.set(
        xlim=(0, model.geometry.length_x * 1e6),
        ylim=(0, model.geometry.length_y * 1e6),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=f"{path.replace('_', ' ').title()} laser trajectory",
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.legend()
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(figure)
