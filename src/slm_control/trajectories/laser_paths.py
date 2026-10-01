"""Laser paths expressed in continuous grid coordinates."""

from __future__ import annotations

import math
from collections.abc import Callable

from slm_control.config import GeometryConfig


def _straight(s: float, nx: int, ny: int) -> tuple[float, float]:
    return 0.2 * nx + 0.3 * nx * s, 0.5 * ny


def _circle(s: float, nx: int, ny: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * s
    return 0.5 * nx + nx / 3 * math.cos(angle), 0.5 * ny + ny / 3 * math.sin(angle)


def _square(s: float, nx: int, ny: int) -> tuple[float, float]:
    if s <= 0.25:
        return 0.25 * nx, (0.25 + 2 * s) * ny
    if s <= 0.5:
        return (0.25 + 2 * (s - 0.25)) * nx, 0.75 * ny
    if s <= 0.75:
        return 0.75 * nx, (0.75 - 2 * (s - 0.5)) * ny
    return (0.75 - 2 * (s - 0.75)) * nx, 0.25 * ny


def _spiral(s: float, nx: int, ny: int) -> tuple[float, float]:
    angle = 4.0 * math.pi * s
    radius = (1.0 - s) / 3.0
    return nx * (0.5 + radius * math.cos(angle)), ny * (0.5 + radius * math.sin(angle))


def _square_spiral(s: float, nx: int, ny: int) -> tuple[float, float]:
    # Piecewise path retained from the original implementation.
    if s < 1 / 6:
        return nx / 4, ny * (1 / 4 + 3 * s)
    if s < 1 / 3:
        return nx * (1 / 4 + 3 * (s - 1 / 6)), 3 * ny / 4
    if s < 1 / 2:
        return 3 * nx / 4, ny * (3 / 4 - 3 * (s - 1 / 3))
    if s < 5 / 8:
        return nx * (3 / 4 - 3 * (s - 1 / 2)), ny / 4
    if s < 3 / 4:
        return 3 * nx / 8, ny * (1 / 4 + 3 * (s - 5 / 8))
    if s < 5 / 6:
        return nx * (3 / 8 + 3 * (s - 3 / 4)), 5 * ny / 8
    if s < 11 / 12:
        return 5 * nx / 8, ny * (5 / 8 - 3 * (s - 5 / 6))
    if s < 23 / 24:
        return nx * (5 / 8 - 3 * (s - 11 / 12)), 3 * ny / 8
    return nx / 2, ny * (3 / 8 + 3 * (s - 23 / 24))


def _zigzag(s: float, nx: int, ny: int) -> tuple[float, float]:
    points = (
        (0.25, 0.25), (0.25, 0.75), (5 / 12, 0.75), (5 / 12, 0.25),
        (7 / 12, 0.25), (7 / 12, 0.75), (0.75, 0.75), (0.75, 0.25),
    )
    scaled = min(s, 1.0) * (len(points) - 1)
    index = min(int(scaled), len(points) - 2)
    fraction = scaled - index
    x0, y0 = points[index]
    x1, y1 = points[index + 1]
    return nx * (x0 + fraction * (x1 - x0)), ny * (y0 + fraction * (y1 - y0))


_PATHS: dict[str, Callable[[float, int, int], tuple[float, float]]] = {
    "straight": _straight,
    "circle": _circle,
    "square": _square,
    "spiral": _spiral,
    "square_spiral": _square_spiral,
    "square spiral": _square_spiral,
    "zigzag": _zigzag,
}


def available_paths() -> tuple[str, ...]:
    return tuple(name for name in _PATHS if " " not in name)


def position(
    name: str, time: float, duration: float, geometry: GeometryConfig
) -> tuple[float, float]:
    """Return laser position in metres, clipped to the printing interval."""
    try:
        path = _PATHS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown path {name!r}; choose from {available_paths()}") from exc
    progress = min(max(float(time) / duration, 0.0), 1.0)
    grid_x, grid_y = path(progress, geometry.cells_x, geometry.cells_y)
    return grid_x * geometry.dx, grid_y * geometry.dy
