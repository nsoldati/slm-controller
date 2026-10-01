# Controllers

## Supported workflows

| Controller | Direct simulation | Training | Checkpoint evaluation |
|---|---:|---:|---:|
| `open_loop` | yes | constant power | yes |
| `output_feedback` | yes | gain per layer/time sample | yes |
| `error_feedback` | yes | single: gain schedule; multi: specialized two-stage method | yes |
| `pi` | yes | proportional and integral gains | yes |

Direct simulation and checkpoint evaluation constrain applied laser power to
`[0, maximum_power]`. Generic training also clamps power. Specialized
multilayer error-feedback training instead leaves its candidate power
unclipped and adds a soft penalty for violations; evaluation of the resulting
controller is clipped to the physical interval.

### Open loop

The same optimized power is applied at every sample and layer. Its YAML initial
value is `controller.power`.

### Output feedback

This controller preserves the legacy single-layer law:

```text
u[layer, time] = gain[layer, time] * measurement[time].
```

Because the gain is independently trainable at every sample, the controller
can retain a time-varying laser-power schedule even at exact reference
tracking. This is necessary to reproduce the near-perfect legacy square-spiral
result. Direct simulation uses the scalar YAML `output_gain`; training produces a
schedule with shape `(layers, samples_per_layer)`.

### Error feedback

The error-feedback controller uses the conventional tracking error

```text
u = feedforward + gain * (reference - measurement).
```

Thus, a positive gain increases laser power below the reference and decreases
it above the reference. Direct simulation and generic single-layer training
use this convention. Generic checkpoints retain it during evaluation. The
specialized multilayer exception is described below.

Direct simulation uses the scalar YAML `gain`. Single-layer generic training
uses the conventional equation above.

For a multi-layer training command, `error_feedback` instead selects the
legacy two-stage research algorithm. Stage one trains
`u[1,t] = K_first[t] y[1,t]`. Its optimized input `u_bias[t]` is then frozen.
Stage two optimizes one shared schedule `K[t]` over every later-layer error:

```text
u[layer,t] = u_bias[t] + K[t] * (measurement[layer,t] - reference[t])
```

This legacy equation deliberately uses `measurement - reference`; it therefore
has the opposite gain sign convention from the generic controller. Feedback is
disabled for samples `0...9` on later layers. The loss covers layers `2...L`,
and autograd propagates state sensitivity through all intervening cooling,
recoating, and ROI transitions. Both `u_bias[t]` and `K[t]` are shared across
the later layers—there is no layer-indexed `K[layer,t]`. A finite,
objective-decreasing acceptance check rejects unstable momentum updates and
reduces the stage-two learning rate before continuing.

### PI

The PI controller uses the conventional error `reference - measurement` and
resets its integral state at the start of every layer. Training optimizes one
proportional gain and one integral gain.

## LQR utility

`controllers/lqr.py` contains a reusable finite-horizon Riccati recursion. It
is not exposed as a trainable CLI controller because an LQR experiment must
explicitly define its state reference and Q/R cost matrices. Historical LQR
experiments remain available under `legacy/`.

## Checkpoint compatibility

Checkpoints record the controller type, single/multi scope, layer count, time
grid, epochs, learning rate, learned parameters, stage losses, and resolved
training configuration. Specialized multilayer checkpoints store
`first_layer_gains`, `bias_power`, and `later_layer_gains` separately.

Evaluation currently validates the controller type, single/multi scope, layer
count, and number of samples per layer. It does not compare the scan path,
reference, geometry, material parameters, optimizer settings, or complete
resolved configuration. Always evaluate with the same experiment YAML used
for training.
