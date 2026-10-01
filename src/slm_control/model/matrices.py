"""Low-level matrix construction without experiment-level dependencies."""

from __future__ import annotations

import torch


def grid_laplacian(
    cells_x: int,
    cells_y: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the graph Laplacian of a rectangular four-neighbour grid.

    Nodes use row-major `(y, x)` ordering. The implementation avoids the
    NetworkX dependency used by the original scripts.
    """
    count = cells_x * cells_y
    adjacency = torch.zeros((count, count), dtype=dtype, device=device)
    for y in range(cells_y):
        for x in range(cells_x):
            node = y * cells_x + x
            if x + 1 < cells_x:
                right = node + 1
                adjacency[node, right] = adjacency[right, node] = 1.0
            if y + 1 < cells_y:
                below = node + cells_x
                adjacency[node, below] = adjacency[below, node] = 1.0
    return torch.diag(adjacency.sum(dim=1)) - adjacency


def block_slice(layer: int, cells_per_layer: int) -> slice:
    start = layer * cells_per_layer
    return slice(start, start + cells_per_layer)

