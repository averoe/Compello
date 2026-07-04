# Compello

Compello is a constraint-driven autotraining framework. It lets you declare the
properties a model should satisfy, turns each declaration into a differentiable
signal, and steers training toward satisfying all of them in real time, across
PyTorch, TensorFlow/Keras, and JAX.



---

## What Compello is

Ordinary training is open-loop. You choose a loss function, a learning rate, and
an epoch count, hit run, and only find out afterward, by inspecting outputs,
running evaluation, or shipping and hitting a production failure, whether the
model actually behaves the way you needed. The properties you cared about
(stay non-negative, do not change under a rotation, keep this token's
probability above a floor, respect a fairness bound) are never expressed to the
optimizer directly. They live in your head and in post-hoc checks.

Compello makes training closed-loop. You state those properties as assertions in
plain terms:

```python
expect(model.output, "> 0")                        # this output should be positive
expect(model, invariant_to=rotate_90)              # prediction shouldn't change under rotation
expect(compello.wrap(logits)[token], "> 0.6")      # this token's probability should exceed 0.6
expect(model.output, parity_across="group")        # fairness-style parity across a group
```

Each assertion is compiled into a smooth, differentiable penalty. A controller
then watches every constraint's violation on each step and adaptively adjusts
how hard the optimizer is pushed toward satisfying it, raising pressure on
constraints that are being violated, and relaxing it on ones that are satisfied
and stable so they stop competing with the primary task loss unnecessarily.

Compello does not replace your training loop. It attaches to what you already
use, raw PyTorch, HuggingFace `Trainer`, PyTorch Lightning, a native
`tf.GradientTape` loop, Keras 3, or a raw JAX step, through hooks, callbacks, or
an explicit function call, and layers the control logic on top.

## How it works

1. Declare. `compello.wrap(model)` returns a transparent proxy, and `expect(...)`
   records a typed assertion against the model's output, a logit, or the model
   itself. Nothing about your model is monkeypatched.
2. Penalize. Each assertion maps to a differentiable penalty shape (hinge,
   finite-difference monotonicity, L2 invariance, probability floor, and so on),
   written once against a small backend-neutral math layer so it runs identically
   on any backend.
3. Control. Every step, the controller reads the (optionally distributed-synced)
   violation, smooths it, and computes the constraint weight via an adaptive PID
   loop, bounded by a dynamic ceiling and protected by a suite of safeguards
   against the failure modes live gradient intervention introduces.
4. Explain. A live telemetry and insight layer reports, in plain language, what
   the controller did and why, and a post-training report quantifies what each
   constraint cost the primary objective.

## Features at a glance

- Assertion DSL: `expect(...)` with type-based dispatch, custom assertion types,
  and an explicit ambiguity error instead of silent guessing.
- Penalty library: range/inequality, monotonicity, invariance, mask-aware
  probability floor, cross-group parity, Lipschitz bound, cross-view
  consistency, and adaptive-sharpness relaxation for non-differentiable metrics
  (soft IoU, soft F1, spectral gate, soft top-k rank).
- Adaptive controller: `fixed`, `linear_ramp`, `adaptive_pid`, and `dual_ascent`
  strategies, with dual-rate EMA smoothing, a dynamic weight ceiling, adaptive
  proxy sharpness with a hysteresis dead-band, a momentum-aware grace window,
  gradient-accumulation freeze, and log-space multiplier stability.
- Gradient surgery: PCGrad-style projection between the task and constraint
  gradients, layer-scoped to bound its cost.
- Backends: numpy reference backend plus PyTorch, TensorFlow/Keras 3, and JAX
  adapters, each optional and imported only when its framework is present.
- Distributed-safe synchronization: one batched collective per step
  (`all_reduce` / `strategy.reduce` / `lax.pmean`).
- Diagnostics and telemetry: live insight stream, online least-squares
  recovery/sensitivity estimates with confidence gating, pre-flight conflict
  detection, noise-aware cold-start, and post-training reports.
- `trainlint`: a static pre-flight linter (standard-library `ast` only) with
  PyTorch, TensorFlow/Keras, and JAX rule sets, a CLI, and a Flake8 plugin.
- Checkpointing: controller state (EMA buffers, counters, sharpness) serialized
  to JSON, numpy, torch, or orbax formats.
- Classical ML: a blockade for non-differentiable estimators plus NODE and
  FT-Transformer differentiable surrogates and a distillation bridge.
- Interoperability: export to Cooper with an explicit account of dropped state.
- Config-driven: a full run's constraint behavior expressed as one YAML/dict.

---

## Table of contents

- [What Compello is](#what-compello-is)
- [How it works](#how-it-works)
- [Features at a glance](#features-at-a-glance)
- [Design principles](#design-principles)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
  - [The declaration model: wrap and expect](#the-declaration-model-wrap-and-expect)
  - [Assertions and penalties](#assertions-and-penalties)
  - [The adaptive controller](#the-adaptive-controller)
  - [Backends and adapters](#backends-and-adapters)
- [Configuration](#configuration)
- [Diagnostics and telemetry](#diagnostics-and-telemetry)
- [trainlint: the static pre-flight linter](#trainlint-the-static-pre-flight-linter)
- [Distributed training](#distributed-training)
- [Checkpointing](#checkpointing)
- [Classical (non-differentiable) machine learning](#classical-non-differentiable-machine-learning)
- [Cooper interoperability](#cooper-interoperability)
- [Known failure modes and resolutions](#known-failure-modes-and-resolutions)
- [Public API overview](#public-api-overview)
- [Worked examples](#worked-examples)
- [Testing](#testing)
- [Verification status](#verification-status)

- [Relation to prior work](#relation-to-prior-work)
- [Project documents](#project-documents)
- [License](#license)

---

## Design principles

1. Declarative correctness. You state what should be true (`expect(...)`), not
   how to penalize it. Compello selects and shapes the penalty.
2. No monkeypatching. Integration happens through documented hook and callback
   APIs, or explicit function calls you make yourself. Compello never rewrites
   the internals of PyTorch, TensorFlow, Keras, or JAX.
3. Optional dependencies, per active backend. The core imports nothing beyond a
   thin backend-interface protocol. A single-backend install pulls in only that
   framework.
4. Portable math, framework-native execution. The assertion DSL and controller
   math are backend-agnostic scalar and array operations. The deep safety
   mechanisms that require framework-specific primitives (distributed
   collectives, backward hooks, optimizer internals) are implemented per backend
   and documented where a guarantee differs.
5. Honest positioning. Where a mechanism is equivalent-in-goal but not
   identical-in-guarantee across backends, that is stated rather than smoothed
   over.

---

## Installation

Compello's core has no hard runtime dependencies. Install the extras for the
backend and features you need.

```
pip install compello                 # core only
pip install compello[numpy]          # numpy reference backend (recommended)
pip install compello[torch]          # PyTorch backend
pip install compello[tensorflow]     # TensorFlow / Keras backend
pip install compello[jax]            # JAX + Optax backend
pip install compello[config]         # YAML config support
pip install compello[dev]            # numpy + pyyaml + pytest for development
```

Requires Python 3.9 or newer.

The numpy backend is a reference and testing backend, not a deep-learning
framework. It lets the controller math, penalties, and diagnostics run and be
validated end-to-end without installing PyTorch, TensorFlow, or JAX.

---

## Quickstart

```python
import numpy as np
import compello
from compello import expect
from compello.controller import Controller, ControllerConfig

# 1. Wrap your model (returns a transparent proxy; the underlying model is
#    unchanged and still callable exactly as before).
model = compello.wrap(my_model)

# 2. Declare correctness properties.
positivity = expect(model.output, "> 0", name="positivity")

# 3. Create a controller and register the assertions.
controller = Controller(ControllerConfig(strategy="adaptive_pid", tolerance=1e-3))
controller.register_assertions([positivity])

# 4. In your training loop: run the forward pass, read the violation, let the
#    controller pick the constraint weight, and add weight * penalty to the loss.
out = model(batch)                       # ordinary forward call
violation = positivity.violation_scalar()
result = controller.step({positivity.name: violation})
weight = controller.states[positivity.name].weight
total_loss = task_loss + weight * positivity.violation()
# ... backward and optimizer step as usual ...
```

For a runnable end-to-end example using the numpy backend (no deep-learning
framework required), see `examples/demo_training.py`.

---

## Core concepts

### The declaration model: wrap and expect

`compello.wrap(obj)` returns a transparent proxy:

- Wrapping a model returns a `ModelProxy` that forwards every attribute access
  and call to the underlying module. It snapshots the most recent forward output
  so `model.output` yields a live `OutputTarget` reflecting the current step.
- Wrapping a tensor returns a `TensorProxy` whose `__getitem__` yields a
  `LogitTarget` (for logit- or probability-level constraints).

`compello.unwrap(proxy)` returns the original object. Wrapping is idempotent.

```python
model = compello.wrap(model)             # ModelProxy
logits = compello.wrap(raw_logits)       # TensorProxy
token_target = logits[target_token]      # LogitTarget
```

### Assertions and penalties

`expect()` dispatches on the type of its target, then on the keyword supplied.
A bare predicate on an `OutputTarget` is a range/inequality; on a `LogitTarget`
it is a probability floor evaluated in probability space.

```python
expect(model.output, lambda y: y > 0)                          # range / inequality
expect(model.output, monotonic_in="age")                       # monotonicity
expect(model, invariant_to=flip_horizontal)                    # invariance
expect(compello.wrap(logits)[token], "> 0.6")                  # probability floor
expect(model.output, parity_across="group")                    # fairness-style parity
expect(model.output, lipschitz_bound=1.0)                      # smoothness
expect(model.output, consistent_across=["view_a", "view_b"])   # cross-view consistency
```

Built-in penalty shapes: hinge range, monotonicity (finite differences),
L2 invariance, mask-aware probability floor, cross-group parity, Lipschitz
bound, cross-view consistency, and adaptive-sharpness sigmoid relaxation for
non-differentiable metric proxies.

If the target is a raw value with no typed wrapper and no explicit
`assertion_type=`, `expect()` raises `AmbiguousAssertionError` at declaration
time rather than guessing. You can register custom assertion types:

```python
from compello import register_assertion_type, math as cmath

def below_cap(tensor, *, cap):
    return cmath.mean(cmath.relu(cmath.asarray(tensor) - cap))

register_assertion_type("below_cap", below_cap)
expect(compello.wrap(x), assertion_type="below_cap", cap=2.0)
```

### The adaptive controller

Instead of a fixed penalty weight, a controller watches violation every step and
adjusts. Strategies (`ControllerConfig(strategy=...)`):

- `fixed`: static weights (a plain multi-term loss baseline).
- `linear_ramp`: schedule-driven weight, ignoring live violation.
- `adaptive_pid` (default): a PID controller on an EMA-smoothed violation
  signal, with a dynamic weight ceiling, adaptive proxy sharpness, and a
  momentum-aware grace window.
- `dual_ascent`: the classic Lagrangian multiplier update, for a grounded
  baseline comparison.

The controller math is expressed as scalar recurrence relations, identical on
every backend; only how the state is stored differs by backend.

### Backends and adapters

The core programs against a small backend-interface protocol
(`compello.math.Backend` for array ops; `compello.backends.protocol.TrainingAdapter`
for training-loop integration). Concrete backends implement it and register
themselves when their library is importable:

- numpy: reference and testing backend.
- PyTorch: `compello.backends.torch_backend` (raw loop, HuggingFace `Trainer`
  callback, PyTorch Lightning callback, DDP/FSDP sync, `torch.compile`-safe
  gradient interception, AMP unscaling, momentum-buffer surgery).
- TensorFlow / Keras 3: `compello.tf` (`ConstraintTape`) and a Keras callback,
  plus a `keras.ops` math backend.
- JAX: `compello.jax` (`init_controller_state`, `steer_step`), `lax.pmean`
  synchronization, static-shape violation virtualization, Optax momentum surgery.

If a backend is not installed, its adapter simply is not imported.

---

## Configuration

A full run's constraint behavior is expressible as a single declarative config
(dict or YAML):

```yaml
constraints:
  - name: positivity
    type: range
    target: model.output
    condition: "> 0"
    initial_weight: 1.0
  - name: token_confidence
    type: probability_floor
    target: logits[target_token]
    condition: "> 0.6"
    respect_loss_mask: true

controller:
  strategy: adaptive_pid
  patience: 500
  max_steps: 50000
  on_plateau: reduce_lr            # or "rollback", "report_infeasible"
  ema_decay: 0.97
  ema_fast_decay: 0.7
  ema_override_steps: 5
  ema_baseline_window: 200
  weight_ceiling: 25.0
  sharpness_hysteresis: 1.5
  sharpness_patience: 0            # >0 enables dead-band re-arm decay
  accumulation_steps: 1            # >1 freezes updates until the macro boundary
  log_space_stability: false       # AMP/FP16 underflow-safe multiplier updates

backend: huggingface_trainer       # lightning, raw_pytorch, tf_gradient_tape, keras3, jax_native
distributed: auto                  # PyTorch/TF auto-detect; JAX requires explicit axis_name

modality: vision                   # text, audio, tabular, multimodal

diagnostics:
  conflict_check: true
  dry_run: true
  holdout_validation: true
  gradient_surgery: true
  gradient_surgery_scope: last_n_layers:8
  telemetry: compact               # verbose, silent
```

Load with `compello.load_config(path_or_dict)`.

---

## Diagnostics and telemetry

Compello surfaces what the controller is doing in plain language, and stays
quiet while metrics are stable.

- `InsightEngine` produces a compact per-step status line and expands into
  boxed insight blocks only on real state transitions (gradient conflict and
  surgery, proxy vanishing-zone and alpha tuning, weight relaxation on recovery,
  ceiling lock, plateau intervention). Numeric projections (for example,
  estimated steps to recovery) come from an online least-squares fit and are
  gated by an R-squared confidence threshold: below 0.5 they carry a
  low-confidence qualifier, below 0.2 they are suppressed and replaced by a
  qualitative statement.
- `SensitivityProfiler` produces a post-training marginal-cost report: how much
  task performance each constraint traded for, with the fit's R-squared shown.
- `non_convergence_report(controller)` explains, for a constraint that did not
  reach tolerance, whether it plateaued at the ceiling (likely infeasible) or
  was still climbing (needs more steps), with a concrete next step.
- `render_capacity_report(...)` composes the full post-training report.

Terminal output degrades gracefully from Unicode/emoji glyphs to ASCII when the
output stream is not UTF-8 (for example, when piped to a file).

---

## trainlint: the static pre-flight linter

`trainlint` catches training-loop logic bugs before a run starts. It uses only
the standard-library `ast` module and never imports or executes your model code,
so it cannot be broken by a missing GPU or a version mismatch. It auto-detects
the active framework by import scanning.

```
trainlint path/to/train.py
trainlint --shield path/to/train.py     # rich, boxed report with fix templates
trainlint --backend pytorch,jax train.py
```

Rule sets:

- PyTorch: detached-loss (memory leak), missing zero_grad, missing train/eval
  mode, missing no_grad in eval, in-place op on a leaf tensor, DataLoader shuffle
  misuse, compiled-optimizer-step conflict, ambiguous assertion.
- TensorFlow / Keras: controller state baked as a constant inside `@tf.function`,
  gradient computation outside a `GradientTape`, non-differentiable op in an
  assertion target, `tf.Variable` created inside a traced function.
- JAX: in-place mutation inside a jitted function, `print`/logging inside a
  jitted body, `donate_argnums` use-after-donation hazard, `distributed: auto`
  under the JAX backend.

`trainlint` is also available as a Flake8 plugin (codes prefixed `TL`).

---

## Distributed training

Local constraint violations are packed into a single array and reduced once per
step (not once per constraint). The batching discipline is backend-agnostic; the
collective differs by backend:

- PyTorch: `torch.distributed.all_reduce(stacked, op=ReduceOp.AVG)`, gated on
  `torch.distributed.is_initialized()`.
- TensorFlow: `tf.distribute.get_strategy().reduce(MEAN, stacked, axis=0)`.
- JAX: `jax.lax.pmean(stacked, axis_name=...)` inside a `pmap`/`jit` region.

JAX has no runtime equivalent of `is_initialized()`, so `distributed: auto` is
invalid under the JAX backend; an explicit `axis_name` is required, and this is
enforced at pre-flight.

---

## Checkpointing

Compello checkpoints the controller-internal state that its adaptive layer needs
to resume without restarting cold: the slow and fast EMA buffers, the rolling
baseline accumulator, stability and plateau counters, and per-constraint
adaptive sharpness with its hysteresis arm.

```python
from compello import save_controller, load_controller

save_controller(controller, "ckpt.json")     # portable JSON
controller = load_controller("ckpt.json", config)
```

Format is chosen by extension: `.json` (portable), `.npz` (numpy),
`.pt` (torch), and orbax PyTree checkpoints for JAX. The controller-state fields
are identical across formats.

---

## Classical (non-differentiable) machine learning

Compello steers models that expose a continuous, per-step gradient interface.
Tree ensembles (Random Forests, XGBoost, LightGBM) and scikit-learn's opaque
`fit()` estimators (including `SGDClassifier`/`SGDRegressor`) do not qualify, and
`compello.check_steerable(model)` raises `UnsupportedTargetError` for them at
pre-flight rather than failing deep inside the controller.

For those problems, use a differentiable surrogate:

```python
from compello import build_node, build_ft_transformer, distillation_bridge

student = build_node(in_features=32, out_features=1, backend="torch")
bridge = distillation_bridge(teacher=xgboost_model, student=student,
                             constraints=my_constraints)
```

`build_node` and `build_ft_transformer` provide Neural Oblivious Decision
Ensemble and FT-Transformer surrogates. Both also ship a pure-numpy reference
forward pass (`NodeReference`, `FTTransformerReference`) usable without a
framework. `distillation_bridge` combines a distillation loss against a trained
teacher with the constraint penalties in the same controller loop. A
differentiable surrogate typically has a lower accuracy ceiling than a tuned
tree ensemble on structured tabular data; whether the trade-off is worth the
behavioral control is a decision for your problem and data.

---

## Cooper interoperability

`compello.export_to_cooper(assertions, controller=...)` describes a hand-off to
Cooper as a snapshot of current multiplier values. Compello-specific state has
no destination in Cooper's object model and is explicitly reported as dropped:
the slow and fast EMA buffers, dual-rate spike detection and rolling baseline,
adaptive sharpness and hysteresis, and the momentum-aware grace window.
`compello.export_to_cooper_objects(...)` constructs live Cooper multiplier
objects when `cooper` is installed, and otherwise returns the description with a
note.

---

## Known failure modes and resolutions

Live gradient intervention introduces failure modes a passive metrics tracker
never faces. Compello documents each with its resolution:

- Distributed multiplier desynchronization: batched cross-replica averaging.
- PID derivative kick from mini-batch noise: dual-rate EMA with a
  standard-deviation-based spike detector and bounded lag override.
- Proxy gradient vanishing: adaptive sharpness with a hysteresis dead-band, plus
  a patience-based re-arm decay so the controller cannot lock up in the band.
- Primary-loss eradication: a dynamic weight ceiling with an explicit
  infeasibility diagnostic instead of unbounded multiplier growth.
- Packed-sequence mask blindness: mask-aware logit constraints.
- Compilation graph breaks and silent hook loss: penalty terms added before
  backward need no hook; gradient surgery runs in the eager gap; intermediate
  activation gradients use a compiler-aware custom op; a pre-flight check flags a
  fully fused optimizer step.
- Stateful optimizer momentum bleed: a momentum-aware grace window by default,
  with opt-in direct momentum-buffer surgery.
- Gradient fusion: un-fused, layer-scoped gradient surgery so task and
  constraint directions are isolated before they blend.
- AMP underflow and scaler interference: unscaled-gradient reads and an
  optional log-space multiplier update.
- Gradient-accumulation poisoning: violations are averaged across micro-batches
  and the multiplier updates only at the macro boundary.

---

## Public API overview

Declaration: `wrap`, `unwrap`, `expect`, `register_assertion_type`,
`OutputTarget`, `LogitTarget`, `ModelTarget`.

Control: `Controller`, `ControllerConfig`.

Diagnostics: `apply_gradient_surgery`, `scoped_gradient_surgery`,
`project_out_conflict`, `detect_conflicts`, `RollingRegressor`,
`ColdStartMonitor`, `DiagnosticsRunner`, `InsightEngine`, `SensitivityProfiler`,
`non_convergence_report`.

Rendering: `render_preflight_shield`, `render_capacity_report`, `Style`.

Validation and config: `validate`, `preflight`, `dry_run`, `load_config`,
`CompelloConfig`.

Checkpointing: `save_controller`, `load_controller`.

Classical ML: `check_steerable`, `is_steerable`, `build_node`,
`build_ft_transformer`, `distillation_bridge`.

Interop: `export_to_cooper`, `export_to_cooper_objects`.

Static analysis: `compello.trainlint.lint_source`, `lint_file`.

---

## Worked examples

Every example marked "runnable" executes on the numpy reference backend with no
deep-learning framework installed. Examples marked with a framework requirement
show the intended integration shape for that backend.

### Example 1: closed-loop training with a constraint (runnable)

A pure-numpy model output is optimized toward a target while a non-negativity
constraint is steered by the adaptive controller. The controller raises the
constraint weight while it is violated and relaxes it once satisfied.

```python
import numpy as np
import compello
from compello import expect
from compello.controller import Controller, ControllerConfig

rng = np.random.default_rng(0)
target = np.abs(rng.normal(0.6, 0.4, size=16))   # non-negative targets (feasible)
theta = rng.normal(0.0, 0.5, size=16)            # the "model output" we optimize

holder = {"out": theta}
model = compello.wrap(lambda _batch=None: holder["out"])   # transparent proxy

pos = expect(model.output, "> 0", name="non_negativity")
ctrl = Controller(ControllerConfig(strategy="adaptive_pid", tolerance=0.02, patience=40))
ctrl.register_assertions([pos])

lr = 0.1
result = None
for step in range(1000):
    holder["out"] = theta
    model()                                       # forward pass refreshes .output
    task_grad = 2.0 * (theta - target) / theta.size
    penalty_grad = np.where(theta < 0, -1.0 / theta.size, 0.0)
    violation = pos.violation_scalar()
    result = ctrl.step({pos.name: violation})
    weight = ctrl.states[pos.name].weight
    theta = theta - lr * (task_grad + weight * penalty_grad)
    if result.should_stop:
        break

print("converged:", result.converged,
      "| violation:", round(pos.violation_scalar(), 4),
      "| weight:", round(ctrl.states[pos.name].weight, 3))
```

### Example 2: LLM logit-level, mask-aware constraint (runnable)

Constrain the probability of a target token to stay above a floor, evaluating
only over real target positions via a loss mask. Works on plain numpy arrays;
the same call shape applies to framework logits.

```python
import numpy as np
import compello
from compello import expect

# (positions, vocab) logits; only the first position is a real target
logits = np.array([[5.0, 0.0, 0.0],
                   [0.0, 0.0, 5.0]])
mask = np.array([1.0, 0.0])

conf = expect(compello.wrap(logits)[0], "> 0.6",
              name="token_confidence", respect_loss_mask=True)
print("masked violation:", round(conf.violation_scalar(mask=mask), 4))   # ~0.0
```

### Example 3: custom assertion type (runnable)

Register a bespoke differentiable penalty and use it through the same DSL.

```python
import numpy as np
from compello import register_assertion_type, wrap, expect, math as cmath

def below_cap(tensor, *, cap):
    # penalize any value above `cap`; 0 when satisfied
    return cmath.mean(cmath.relu(cmath.asarray(tensor) - cap))

register_assertion_type("below_cap", below_cap)

a = expect(wrap(np.array([1.0, 5.0])), assertion_type="below_cap", cap=2.0, name="cap")
print("violation:", a.violation_scalar(cap=2.0))   # mean(relu([-1, 3])) = 1.5
```

### Example 4: layer-scoped gradient surgery (runnable)

Project the conflicting component out of the constraint gradient, restricted to
a subset of parameters so the cost is bounded.

```python
import numpy as np
from compello.diagnostics import scoped_gradient_surgery

task = {"encoder.w": np.array([1.0, 0.0]), "head.w": np.array([1.0, 0.0])}
constraint = {"encoder.w": np.array([-0.5, 1.0]), "head.w": np.array([-0.5, 1.0])}

res = scoped_gradient_surgery(task, constraint, scope="last_n_layers:1")
print("projected:", res.projected, "| cost fraction:", round(res.cost_fraction, 2))
# 'head.w' is orthogonalized against the task gradient; 'encoder.w' is untouched
print("head dot task:",
      round(float(np.dot(res.corrected["head.w"], task["head.w"])), 6))   # ~0.0
```

### Example 5: live telemetry and insights (runnable)

The insight engine emits a compact status line and expands into a boxed insight
block only on real events (here, a reported gradient conflict). Glyphs degrade to
ASCII automatically when the stream is not UTF-8.

```python
from compello.controller import Controller, ControllerConfig
from compello.insights import InsightEngine

ctrl = Controller(ControllerConfig(tolerance=0.02, patience=5))
ctrl.register("spatial_iou", 1.0)
engine = InsightEngine(ctrl, telemetry="compact", total_steps=1000,
                       modality="vision", backend="JAX/XLA")

r = ctrl.step({"spatial_iou": 0.3})
out = engine.observe(
    r, loss=0.14,
    grad_conflicts={"spatial_iou": {"cosine": -0.87, "projected": True,
                                    "layer": "decoder.block.11", "task_loss": "dice_loss"}},
)
print(out.render())
```

Sample output:

```
Step 0/1000 | 0% | [JAX/XLA] | loss: 0.14 | [X] 0/1 Bounds Compliant
  [!] [Compello Runtime Insight - Vision Modality]
  Constraint Target: constraint 'spatial_iou'
  Current Status: violation 0.3 detected (weight now 1.03).
  [Issue Intercepted]: vector conflict between 'dice_loss' and constraint
    'spatial_iou' (cosine similarity: -0.87 at decoder.block.11).
  * CONTROL SYSTEM INTERVENTIONS EXECUTED:
   1. Gradient Surgery active. Conflicting components projected out at decoder.block.11.
```

### Example 6: post-training sensitivity and non-convergence reports (runnable)

```python
from compello.controller import Controller, ControllerConfig
from compello.reports import SensitivityProfiler, render_capacity_report

ctrl = Controller(ControllerConfig(tolerance=0.02, weight_ceiling=1e9))
ctrl.register("fairness", 1.0)
profiler = SensitivityProfiler(high_impact_threshold=0.01)

for i in range(120):
    ctrl.step({"fairness": 0.5})
    w = ctrl.states["fairness"].weight
    profiler.observe("fairness", weight=w, violation=0.5, task_metric=0.1 + 0.3 * w)

print(render_capacity_report(ctrl, profiler, converged=False,
                             primitive_labels={"fairness": "Fairness Primitive"}))
```

### Example 7: config-driven setup (runnable)

```python
from compello import load_config

cfg = load_config({
    "constraints": [
        {"name": "positivity", "type": "range", "target": "model.output", "condition": "> 0"},
    ],
    "controller": {"strategy": "adaptive_pid", "patience": 500, "weight_ceiling": 25.0},
    "backend": "raw_pytorch",
    "modality": "tabular",
})
print(cfg.controller.strategy, cfg.backend, [c.name for c in cfg.constraints])
```

### Example 8: checkpoint and resume (runnable)

```python
from compello import save_controller, load_controller
from compello.controller import Controller, ControllerConfig

cfg = ControllerConfig(tolerance=0.02, patience=5)
ctrl = Controller(cfg)
ctrl.register("c", 1.0)
for _ in range(30):
    ctrl.step({"c": 0.4})

save_controller(ctrl, "run.json")           # EMA buffers, sharpness, counters preserved
resumed = load_controller("run.json", cfg)  # resumes warm, not cold
assert resumed.weights == ctrl.weights
```

### Example 9: pre-flight linting (command line)

```
$ trainlint --shield train.py
================================================================================
                       COMPELLO PRE-FLIGHT STATIC SHIELD
================================================================================
[X] STATIC CONFIGURATION ERROR DETECTED
-> Triggered by: CompiledOptimizerStepConflictError
[Detail]: optimizer step appears compiled inside the same region as backward ...
[!] WHY THIS WILL HURT YOUR RUN:
AOTAutograd bakes the backward pass into a static graph; gradient surgery needs
an eager gap between backward and the optimizer step ...
[fix] ACTIONABLE FIX:
Split the compiled region: compile forward+backward, keep optimizer.step() eager.
```

### Example 10: PyTorch raw loop (requires `compello[torch]`)

Un-fused, layer-scoped gradient surgery writes corrected gradients in place, and
a momentum-aware grace window is opened for the constraint after surgery engages.

```python
# requires: pip install compello[torch]
import compello
from compello import expect
from compello.controller import Controller, ControllerConfig
from compello.backends.torch_backend import TorchAdapter

model = compello.wrap(net)
adapter = TorchAdapter(model, optimizer)
positivity = expect(model.output, "> 0", name="positivity")
ctrl = Controller(ControllerConfig()); ctrl.register_assertions([positivity])

for batch, y in loader:
    optimizer.zero_grad()
    out = model(batch)
    task_loss = criterion(out, y)
    weight = ctrl.states["positivity"].weight
    constraint_loss = weight * positivity.violation()

    # independent task/constraint backward paths, restricted to the last 8 layers
    adapter.unfused_scoped_surgery(task_loss, constraint_loss, scope="last_n_layers:8")

    ctrl.engage_surgery("positivity", adapter.optimizer_beta1())  # grace window
    ctrl.step({"positivity": positivity.violation_scalar()})
    optimizer.step()
```

### Example 11: JAX step function (requires `compello[jax]`)

Controller state is an explicit PyTree threaded through the jitted step; the
distributed reduction is a single `pmean` over a named axis.

```python
# requires: pip install compello[jax]
import jax, optax
import compello.jax as cjax

state = cjax.init_controller_state(constraints=my_constraints)

@jax.jit
def train_step(params, opt_state, controller_state, batch):
    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    steered_grads, controller_state = cjax.steer_step(
        grads, preds, controller_state, constraints=my_constraints, axis_name="batch")
    updates, opt_state = optimizer.update(steered_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, controller_state
```

### Example 12: TensorFlow ConstraintTape (requires `compello[tensorflow]`)

```python
# requires: pip install compello[tensorflow]
import compello
from compello.tf import ConstraintTape, expect
from compello.backends.tf_backend import TFAdapter

model = compello.wrap(keras_model)
adapter = TFAdapter(model, optimizer)
expect(model.output, type="range", condition="> 0")

with ConstraintTape(controller=ctrl, assertions=assertions, adapter=adapter) as tape:
    preds = model(x, training=True)
    task_loss = compute_loss(y, preds)

steered = tape.steer_gradients(primary_loss=task_loss, variables=model.trainable_variables)
optimizer.apply_gradients(zip(steered, model.trainable_variables))
```

### Example 13: classical ML surrogate + distillation

The numpy reference forward is runnable now; the torch builder and a real
teacher require `compello[torch]` and your gradient-boosting library.

```python
import numpy as np
from compello import NodeReference, build_node, distillation_bridge, expect, wrap

# runnable: pure-numpy NODE reference forward
student_ref = NodeReference(in_features=8, out_features=1, n_trees=4, depth=3)
print(student_ref.forward(np.zeros((2, 8))).shape)   # (2, 1)

# framework path (requires torch + a trained teacher):
# student = build_node(in_features=8, out_features=1, backend="torch")
# fairness = expect(wrap(student_output), parity_across="group", name="parity")
# bridge = distillation_bridge(teacher=xgboost_model, student=student,
#                              constraints=[fairness])
```

---

## Testing

```
pip install compello[dev]
pytest -q
```

The suite runs without any deep-learning framework installed, using the numpy
reference backend.

---

## Verification status

Compello is honest about what is validated and what is not.

- Framework-independent core (assertion DSL, controller, EMA/PID, sharpness and
  hysteresis, weight ceiling, gradient-accumulation freeze, log-space stability,
  diagnostics, online regression, insight engine, reports, checkpoint
  serialization, trainlint, config, validation, Cooper export description,
  classical-ML surrogate reference forwards): covered by the automated test
  suite and exercised end-to-end on the numpy backend.
- Backend-agnostic kernels extracted from the adapters (PCGrad projection,
  batched-sync discipline, layer scope selection): covered by the test suite.
- Framework adapter paths validated against the real frameworks on CPU,
  single-device (see `tests/integration`, run with each framework installed):
  the torch/tf/jax math backends; `wrap`/`register_forward_hook` capture;
  gradient-scope reads; un-fused surgery via `torch.func.vjp` and independent
  backward paths; `functional_vjp_grads`; direct momentum-buffer projection
  (`exp_avg`, Keras `optimizer.variables`, Optax `opt_state.mu`); AMP GradScaler
  unscale paths; the `torch.library.custom_op` gradient interceptor; the batched
  distributed collective via a single-rank gloo group and `jax.lax.pmean` under
  `pmap`; static-shape JAX violation virtualization; the TensorFlow
  `ConstraintTape` surgery path; and the `keras.ops` penalty backend. Validated
  versions: torch 2.12, jax 0.10 / optax 0.2, tensorflow 2.21 / keras 3.15,
  Python 3.13.
- Still not validated on hardware (tracked on the roadmap): multi-process
  distributed (only single-rank collectives were exercised), graph interception
  under actual `torch.compile`/`tf.function` compilation, the HuggingFace
  `Trainer` and PyTorch Lightning callbacks against those libraries, and orbax
  PyTree checkpoint serialization.

The integration tests skip automatically when a framework is absent, so the
default suite stays framework-free. A bug found during this validation (the
TensorFlow `ConstraintTape` computed its constraint penalty after the tape
context closed, so the constraint gradient was silently dropped) has been fixed
and is covered by a regression test.



---

## Relation to prior work

The mathematical foundation, Lagrangian-style constrained optimization with
adaptive multipliers, is established. Google's TFCO implemented it for
TensorFlow (TF1 graph mode, now deprecated) and Cooper provides an actively
maintained PyTorch implementation. Compello builds on that foundation and adds
the surrounding engineering: an assertion DSL, adaptive control with documented
failure-mode resolutions, diagnostics and telemetry, distributed-training
safety, a static pre-flight linter, and multi-backend reach. The PCGrad-style
projection used in gradient surgery is from Yu et al.; Compello applies it
between a task loss and a live, adaptively-weighted constraint gradient.

---

## Project documents

- `CONTRIBUTING.md`: development setup, coding standards, and the pull-request
  workflow.
- `SECURITY.md`: the vulnerability disclosure policy and security notes on
  config, checkpoints, and custom assertion code.
- `CHANGELOG.md`: notable changes per release (Keep a Changelog format).
- `examples/`: runnable end-to-end examples, including `demo_training.py`.
- `tests/`: the automated suite; `tests/integration/` holds the framework
  adapter tests that run when a backend is installed.

---

## License

Apache License 2.0. See `LICENSE`.
