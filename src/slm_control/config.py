"""Typed, serializable configuration for SLM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GeometryConfig:
    length_x: float = 500e-6
    length_y: float = 500e-6
    layer_thickness: float = 50e-6
    cells_x: int = 25
    cells_y: int = 25

    @property
    def cell_count(self) -> int:
        return self.cells_x * self.cells_y

    @property
    def dx(self) -> float:
        return self.length_x / self.cells_x

    @property
    def dy(self) -> float:
        return self.length_y / self.cells_y


@dataclass(frozen=True)
class MaterialConfig:
    substrate_temperature: float = 900.0
    ambient_temperature: float = 300.0
    porosity: float = 0.6
    volumetric_heat_capacity: float = 4.25e6
    powder_conductivity: float = 0.5
    dense_conductivity: float = 20.0
    convection_coefficient: float = 10.0


@dataclass(frozen=True)
class LaserConfig:
    maximum_power: float = 50.0
    absorptivity: float = 0.42
    beam_radius: float = 60e-6


@dataclass(frozen=True)
class TimeConfig:
    printing_duration: float = 1.25e-3
    recoating_duration: float = 1.25e-3
    step_size: float = 10e-6

    @property
    def steps(self) -> int:
        return round(self.printing_duration / self.step_size) + 1


@dataclass(frozen=True)
class ModelConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    roi_layers: int = 4
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.geometry.cells_x < 2 or self.geometry.cells_y < 2:
            raise ValueError("The thermal grid needs at least two cells in each direction")
        if not 0.0 <= self.material.porosity < 1.0:
            raise ValueError("Porosity must lie in [0, 1)")
        if self.time.step_size <= 0 or self.time.printing_duration <= 0:
            raise ValueError("Simulation durations and step size must be positive")
        if self.roi_layers < 1:
            raise ValueError("roi_layers must be positive")


@dataclass(frozen=True)
class ControllerConfig:
    kind: str = "open_loop"
    power: float = 8.8
    feedforward: float = 8.8
    output_gain: float = 0.005
    gain: float = 0.0
    integral_gain: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    layers: int = 1
    path: str = "square_spiral"
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    reference_temperature: float = 600.0
    measurement_noise_std: float = 0.0
    seed: int = 7
    output_root: str = "results"
    figure_root: str = "figures"

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ValueError("An experiment must contain at least one layer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_from_dict(values: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        geometry=GeometryConfig(**values.get("geometry", {})),
        material=MaterialConfig(**values.get("material", {})),
        laser=LaserConfig(**values.get("laser", {})),
        time=TimeConfig(**values.get("time", {})),
        roi_layers=values.get("roi_layers", 4),
        dtype=values.get("dtype", "float64"),
        device=values.get("device", "cpu"),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment and resolve an optional external model YAML file."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    model_value = raw.pop("model", {})
    if isinstance(model_value, str):
        model_path = (config_path.parent / model_value).resolve()
        model_value = yaml.safe_load(model_path.read_text()) or {}
    return ExperimentConfig(
        model=_model_from_dict(model_value),
        controller=ControllerConfig(**raw.pop("controller", {})),
        **raw,
    )
