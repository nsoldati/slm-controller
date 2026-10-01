"""Finite-horizon discrete linear-quadratic regulator utilities."""

from __future__ import annotations

import torch


def finite_horizon_gains(
    system: torch.Tensor,
    inputs: torch.Tensor,
    state_cost: torch.Tensor,
    input_cost: torch.Tensor,
    terminal_cost: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return gains K[t] for x[t+1] = A x[t] + B[t] u[t]."""
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (time, state, input)")
    horizon, states, controls = inputs.shape
    if system.shape != (states, states):
        raise ValueError("system matrix and input dimensions do not agree")
    value = state_cost.clone() if terminal_cost is None else terminal_cost.clone()
    gains = torch.empty(
        (horizon, controls, states), dtype=system.dtype, device=system.device
    )
    for index in range(horizon - 1, -1, -1):
        b = inputs[index]
        denominator = input_cost + b.T @ value @ b
        gain = torch.linalg.solve(denominator, b.T @ value @ system)
        gains[index] = gain
        value = state_cost + system.T @ value @ (system - b @ gain)
    return gains

