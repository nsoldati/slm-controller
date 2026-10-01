# Experiment protocol

## Recommended sequence

1. Copy `configs/experiments/single_layer.yaml` or `multilayer.yaml`.
2. Give the experiment a unique `name`.
3. Select the scan `path`, number of `layers`, and reference temperature.
4. Record model changes in a separate model YAML if needed.
5. Train with `slm-train` and retain its checkpoint.
6. Evaluate with `slm-evaluate` using matching arguments.
7. Archive the checkpoint and timestamped results directory together.

Use `small_test.yaml` only to verify software changes and commands.

For multi-layer `error_feedback`, pass and record both stage configurations:
`--first-layer-epochs`, `--first-layer-learning-rate`, `--epochs`, and
`--learning-rate`. The recommended legacy square-spiral settings are
respectively 120, `8e-9`, 75, and `3e-6`. Pass them explicitly because the
general CLI defaults for `--epochs` and `--learning-rate` are 100 and `1e-6`.
Also record any change to `--feedback-start` (default 10) or
`--constraint-penalty` (default 100).

## Reproducibility checklist

Every reported experiment should retain:

- experiment and model YAML files;
- trained checkpoint and training-loss history;
- package version and Git revision;
- random seed and measurement-noise standard deviation;
- grid size, time step, number of layers, and ROI size;
- evaluation `metrics.json` and `history.pt` from `results/`;
- the corresponding tracking plot from `figures/evaluations/`;
- any additional constraint or robustness metrics used in the report.

Run `uv run pytest` before producing final results.
