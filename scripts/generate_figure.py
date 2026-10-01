from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slm_control.simulation import SimulationResult
from slm_control.visualization import plot_tracking


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate a tracking figure from a saved run")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values = torch.load(args.history, map_location="cpu", weights_only=True)
    result = SimulationResult(**values)
    output = args.output or Path("figures/evaluations") / f"{args.history.parent.name}.png"
    plot_tracking(result, output)
    print(f"Figure saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
