"""Persistence helpers for reproducible experiment runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from slm_control.config import ExperimentConfig
from slm_control.simulation import SimulationResult


def save_run(experiment: ExperimentConfig, result: SimulationResult) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_directory = Path(experiment.output_root) / f"{timestamp}_{experiment.name}"
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config.yaml").write_text(
        yaml.safe_dump(experiment.to_dict(), sort_keys=False), encoding="utf-8"
    )
    metrics = {
        "rmse_kelvin": result.rmse,
        "layers": experiment.layers,
        "samples": len(result.time),
        "seed": experiment.seed,
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "time": result.time.cpu(),
            "output": result.output.cpu(),
            "power": result.power.cpu(),
            "reference": result.reference.cpu(),
            "state": result.state.cpu(),
            "layer": result.layer.cpu(),
        },
        run_directory / "history.pt",
    )
    return run_directory

