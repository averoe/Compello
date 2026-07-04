# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning. Dates are in ISO 8601 (YYYY-MM-DD).

## [Unreleased]

### Planned

- Framework integration test matrix (PyTorch, TensorFlow/Keras, JAX) covering
  distributed collectives, `torch.compile`/`tf.function` interception, and
  `jit`/`pmap`, to move the framework-specific adapter paths from
  reviewed-and-byte-compiled to hardware-verified.
- Generated-text-level LLM constraints (policy-gradient and preference methods)
  as a distinct capability tier.
- Expanded vision/audio/multimodal modality presets across backends.

## [0.1.0] - 2026-07-05

Initial public release.

### Added

- Declaration model: `compello.wrap`/`unwrap` transparent proxies (`ModelProxy`,
  `TensorProxy`) and typed targets (`OutputTarget`, `LogitTarget`, `ModelTarget`).
- Assertion DSL: `expect(...)` with type-based dispatch, custom assertion types
  via `register_assertion_type`, and `AmbiguousAssertionError` for un-typed
  targets without an explicit `assertion_type`.
- Penalty library: hinge range, monotonicity, L2 invariance, mask-aware
  probability floor, cross-group parity, Lipschitz bound, cross-view
  consistency, and adaptive-sharpness sigmoid relaxation.
- Modality relaxations: soft IoU, soft F1, spectral gate, soft top-k rank.
- Adaptive controller with `fixed`, `linear_ramp`, `adaptive_pid`, and
  `dual_ascent` strategies.
- Control-loop safety mechanisms: dual-rate EMA smoothing with a
  standard-deviation-based spike detector, adaptive proxy sharpness with a
  hysteresis dead-band and patience-based re-arm decay, dynamic weight ceiling
  with infeasibility reporting, momentum-aware grace window, gradient-
  accumulation freeze, and log-space multiplier stability.
- `compello.math` backend dispatcher with numpy reference backend and a
  `keras.ops` backend; thin backend-interface protocol in `compello.math` and
  `compello.backends.protocol`.
- Backend adapters: PyTorch (raw loop, HuggingFace `Trainer` callback, PyTorch
  Lightning callback, DDP/FSDP batched sync, `torch.compile`-safe gradient
  interception via `torch.library.custom_op`, `torch.func.vjp` un-fused scoped
  surgery, AMP GradScaler unscaling, `exp_avg` momentum surgery); TensorFlow
  (`ConstraintTape`, Keras callback, `strategy.reduce` sync, `optimizer.variables`
  momentum surgery); JAX (`init_controller_state`/`steer_step`, `lax.pmean` sync,
  static-shape violation virtualization with `jnp.where`, Optax `opt_state.mu`
  momentum surgery).
- Backend-agnostic kernels extracted from adapters and tested with numpy:
  PCGrad projection (`project_out_conflict`), batched distributed-sync discipline
  (`batched_sync`), and layer scope selection.
- Diagnostics: gradient-surgery cosine and projection, layer-scoped surgery with
  a full-model cost warning, online least-squares recovery/sensitivity estimator
  with R-squared confidence gating, pre-flight conflict detection, noise-aware
  cold-start monitor, and a diagnostics stride runner.
- Telemetry and insight engine: transition-gated boxed insight blocks, compact
  status stream, and Unicode/ASCII glyph auto-degradation.
- Post-training reports: sensitivity/marginal-cost profiler and non-convergence
  diagnostic report; rich pre-flight shield renderer.
- `trainlint` static linter with PyTorch, TensorFlow/Keras, and JAX rule sets, a
  CLI (`trainlint`, `--shield`, `--ascii`), and a Flake8 plugin.
- Checkpoint serialization for controller state: portable JSON and numpy `.npz`
  (tested), with guarded `.pt` and orbax paths.
- Held-out validation (`validate`), pre-flight checks (`preflight`), dry-run
  feasibility (`dry_run`), and declarative YAML/dict config (`load_config`).
- Classical-ML support: `check_steerable`/`is_steerable` blockade for
  non-differentiable estimators, NODE and FT-Transformer surrogates with pure-
  numpy reference forwards and guarded torch builders, and `distillation_bridge`.
- Cooper interoperability: `export_to_cooper` (lossy description with explicit
  dropped-state reporting) and `export_to_cooper_objects` (guarded live objects).
- Packaging: `py.typed` marker; zero-hard-dependency core with per-backend
  optional extras.

### Notes

- The framework-specific adapter paths listed above are complete and byte-
  compiled but are not yet validated on hardware with the frameworks installed.
  See the Verification status section of the README.
