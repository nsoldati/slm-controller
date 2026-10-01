"""Thermal state-space model for one or more SLM layers."""

from __future__ import annotations

import math

import torch

from slm_control.config import ModelConfig
from slm_control.model.matrices import block_slice, grid_laplacian
from slm_control.trajectories import position

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class ThermalModel:
    """Construct continuous-time model matrices from explicit configuration.

    Layer index zero is always the newest/top powder layer. Older layers are
    dense and follow beneath it. This convention is used throughout the new
    package.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            self.dtype = _DTYPES[config.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype {config.dtype!r}") from exc
        self.device = torch.device(config.device)
        self.cells = config.geometry.cell_count
        self._laplacian = grid_laplacian(
            config.geometry.cells_x,
            config.geometry.cells_y,
            dtype=self.dtype,
            device=self.device,
        )
        self._capacity_cache: dict[int, torch.Tensor] = {}
        self._conductance_cache: dict[int, torch.Tensor] = {}
        self._system_cache: dict[int, torch.Tensor] = {}
        self._disturbance_cache: dict[int, torch.Tensor] = {}
        self._cooling_cache: dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def powder_conductance(self) -> float:
        return self.config.material.powder_conductivity * self.config.geometry.dx

    @property
    def dense_conductance(self) -> float:
        return self.config.material.dense_conductivity * self.config.geometry.dy

    @property
    def interface_conductance(self) -> float:
        return 0.5 * (self.powder_conductance + self.dense_conductance)

    @property
    def convection_conductance(self) -> float:
        geometry = self.config.geometry
        return self.config.material.convection_coefficient * geometry.dx * geometry.dy

    @property
    def cell_heat_capacity(self) -> float:
        geometry = self.config.geometry
        return (
            self.config.material.volumetric_heat_capacity
            * geometry.dx
            * geometry.dy
            * geometry.layer_thickness
        )

    def heat_capacity_inverse(self, layers: int) -> torch.Tensor:
        if layers in self._capacity_cache:
            return self._capacity_cache[layers]
        values = torch.full(
            (layers * self.cells,),
            1.0 / self.cell_heat_capacity,
            dtype=self.dtype,
            device=self.device,
        )
        values[: self.cells] /= 1.0 - self.config.material.porosity
        result = torch.diag(values)
        self._capacity_cache[layers] = result
        return result

    def conductance_matrix(self, layers: int) -> torch.Tensor:
        """Return the dimensional thermal conductance matrix before C^-1."""
        if layers < 1:
            raise ValueError("layers must be positive")
        if layers in self._conductance_cache:
            return self._conductance_cache[layers]
        size = layers * self.cells
        matrix = torch.zeros((size, size), dtype=self.dtype, device=self.device)
        identity = torch.eye(self.cells, dtype=self.dtype, device=self.device)

        for layer in range(layers):
            current = block_slice(layer, self.cells)
            lateral = self.powder_conductance if layer == 0 else self.dense_conductance
            losses = self.convection_conductance if layer == 0 else 0.0

            if layers == 1:
                losses += self.interface_conductance
            else:
                if layer > 0:
                    losses += self.interface_conductance if layer == 1 else self.dense_conductance
                if layer + 1 < layers:
                    losses += self.interface_conductance if layer == 0 else self.dense_conductance
                else:
                    losses += self.dense_conductance

            matrix[current, current] = -lateral * self._laplacian - losses * identity

        for upper in range(layers - 1):
            lower = upper + 1
            coupling = self.interface_conductance if upper == 0 else self.dense_conductance
            upper_slice = block_slice(upper, self.cells)
            lower_slice = block_slice(lower, self.cells)
            matrix[upper_slice, lower_slice] = coupling * identity
            matrix[lower_slice, upper_slice] = coupling * identity
        self._conductance_cache[layers] = matrix
        return matrix

    def system_matrix(self, layers: int) -> torch.Tensor:
        if layers not in self._system_cache:
            self._system_cache[layers] = (
                self.heat_capacity_inverse(layers) @ self.conductance_matrix(layers)
            )
        return self._system_cache[layers]

    def disturbance(self, layers: int) -> torch.Tensor:
        """Return C^-1 d for ambient and substrate boundary temperatures."""
        if layers in self._disturbance_cache:
            return self._disturbance_cache[layers]
        physical = torch.zeros(layers * self.cells, dtype=self.dtype, device=self.device)
        material = self.config.material
        physical[: self.cells] = self.convection_conductance * material.ambient_temperature
        substrate_coupling = self.interface_conductance if layers == 1 else self.dense_conductance
        physical[-self.cells :] += substrate_coupling * material.substrate_temperature
        result = self.heat_capacity_inverse(layers) @ physical
        self._disturbance_cache[layers] = result
        return result

    def cooling_map(self, layers: int, duration: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(transition, affine_offset)` for zero-input cooling."""
        key = (layers, float(duration))
        if key not in self._cooling_cache:
            matrix = self.system_matrix(layers)
            transition = torch.matrix_exp(matrix * duration)
            identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
            affine = torch.linalg.solve(
                matrix, (transition - identity) @ self.disturbance(layers)
            )
            self._cooling_cache[key] = transition, affine
        return self._cooling_cache[key]

    def laser_profile(self, path: str, time: float) -> torch.Tensor:
        """Return the physical Gaussian heat-flux distribution over one layer."""
        geometry = self.config.geometry
        laser = self.config.laser
        laser_x, laser_y = position(path, time, self.config.time.printing_duration, geometry)
        x = torch.arange(geometry.cells_x, dtype=self.dtype, device=self.device) * geometry.dx
        y = torch.arange(geometry.cells_y, dtype=self.dtype, device=self.device) * geometry.dy
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        distance_squared = (grid_x - laser_x) ** 2 + (grid_y - laser_y) ** 2
        variance = laser.beam_radius**2 / 9.0
        amplitude = geometry.dx * geometry.dy * laser.absorptivity / (2.0 * math.pi * variance)
        return (amplitude * torch.exp(-0.5 * distance_squared / variance)).reshape(-1)

    def input_vector(self, layers: int, path: str, time: float) -> torch.Tensor:
        physical = torch.zeros(layers * self.cells, dtype=self.dtype, device=self.device)
        physical[: self.cells] = self.laser_profile(path, time)
        return self.heat_capacity_inverse(layers) @ physical

    def observation_vector(self, layers: int, path: str, time: float) -> torch.Tensor:
        """Return the moving spatial weighting retained from the legacy model."""
        vector = torch.zeros(layers * self.cells, dtype=self.dtype, device=self.device)
        vector[: self.cells] = self.laser_profile(path, time)
        return vector

    def initial_state(self, layers: int = 1) -> torch.Tensor:
        return torch.full(
            (layers * self.cells,),
            self.config.material.substrate_temperature,
            dtype=self.dtype,
            device=self.device,
        )
