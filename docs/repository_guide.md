# Repository guide

The repository separates reusable scientific code from experiment definitions
and generated outputs.

## Source package

`src/slm_control` is the only importable implementation. Core modules do not
load checkpoints, parse command-line arguments, show figures, or import
experiment scripts. Dependencies point inward in this order:

1. configuration and trajectories;
2. matrix construction and the thermal model;
3. controllers;
4. simulation and training workflows;
5. persistence, visualization, and command-line interfaces.

The public command-line workflows live in `slm_control.cli`. The small files in
`scripts/` call those same functions, so commands and scripts cannot silently
diverge.

This ordering prevents the circular import between `parameters.py` and
`multiple_layer_controllers.py` in the legacy code.

## Conventions

- SI units are used internally.
- Layer zero is the newest/top powder layer.
- State vectors concatenate layers from top to bottom.
- A state history has shape `(time, state)`.
- Experiment choices belong in YAML rather than module-level constants.
- Generated results do not belong in `src/`.
- Numerical artifacts belong in `results/`; images belong in a category under
  `figures/`, such as `figures/trajectories/` or `figures/evaluations/`.

## Legacy implementation

`legacy/` is a frozen recovery snapshot of the code and outputs that existed
before the refactor. Its scripts retain their original relative paths and are
not imported by the new package. Run them from inside `legacy/` if historical
reproduction is needed. They are expected to retain their original dependency
and circular-import limitations.

## Adding an experiment

Copy a YAML file from `configs/experiments`, change only the parameters needed
for the new study, and run the appropriate CLI. Commit the YAML manifest.
Generated `results/` content—including `metrics.json`—is ignored by Git, so
archive it with the checkpoint and figures outside the repository, add a
selected metric file explicitly with `git add -f`, or copy the reported summary
into a deliberately tracked research record. Large tensor histories should
normally remain outside version control.
