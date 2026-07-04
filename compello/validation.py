"""Held-out validation and pre-flight checks (Sections 6.1, 6.4, 6.5).

- ``validate`` is a *test set for behaviour*: it checks whether constraint
  satisfaction generalises to held-out data, the same way a test-set metric
  checks task accuracy -- a distinct claim from "the controller says this
  converged during training."
- ``preflight`` bundles the static conflict check (6.1) with the named
  backend-configuration checks (the JAX ``distributed: auto`` case in 4.1b and
  the compiled-optimizer-step conflict in 4.6).
- ``dry_run`` (6.5) runs a short trial to see whether violation trends downward
  under aggressive weighting before the full compute budget is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .assertions import Assertion
from .diagnostics.conflict import ConflictReport, detect_conflicts
from .exceptions import (
    CompiledOptimizerStepConflictError,
    DistributedConfigError,
)
from .proxy import ModelProxy, wrap


# --------------------------------------------------------------------------
# Held-out validation (6.4)
# --------------------------------------------------------------------------

@dataclass
class ConstraintValidation:
    name: str
    satisfied_fraction: float
    mean_violation: float
    max_violation: float
    n: int

    @property
    def passed(self) -> bool:
        return self.satisfied_fraction >= 1.0


@dataclass
class ValidationReport:
    per_constraint: Dict[str, ConstraintValidation]
    tolerance: float

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.per_constraint.values())

    def __repr__(self) -> str:
        lines = [f"ValidationReport(tol={self.tolerance}, all_passed={self.all_passed})"]
        for c in self.per_constraint.values():
            lines.append(
                f"  {c.name}: {c.satisfied_fraction*100:.1f}% satisfied "
                f"(mean={c.mean_violation:.4g}, max={c.max_violation:.4g}, n={c.n})"
            )
        return "\n".join(lines)


def validate(
    model: Any = None,
    holdout_data: Optional[Iterable[Any]] = None,
    *,
    constraints: Sequence[Assertion],
    tolerance: float = 1e-3,
    input_key: Optional[str] = None,
) -> ValidationReport:
    """Evaluate ``constraints`` on ``holdout_data`` (Section 6.4).

    Two modes:
    - With ``model``: each item in ``holdout_data`` is fed through the model
      (wrapped if needed) and constraint violations are read from the live
      output. ``input_key`` optionally selects the model input from a dict batch.
    - Without ``model``: each item in ``holdout_data`` is treated as a mapping of
      keyword inputs passed straight to ``assertion.violation(**item)``.
    """
    if holdout_data is None:
        raise ValueError("holdout_data is required")

    if model is not None and not isinstance(model, ModelProxy):
        model = wrap(model)

    acc: Dict[str, List[float]] = {c.name: [] for c in constraints}

    for item in holdout_data:
        inputs: Dict[str, Any] = {}
        if model is not None:
            model_input = item[input_key] if input_key is not None else item
            model(model_input)
            if isinstance(item, dict):
                inputs = {k: v for k, v in item.items() if k != input_key}
        else:
            inputs = dict(item) if isinstance(item, dict) else {}
        for c in constraints:
            acc[c.name].append(float(c.violation_scalar(**inputs)))

    per: Dict[str, ConstraintValidation] = {}
    for c in constraints:
        vals = acc[c.name]
        n = len(vals)
        sat = sum(1 for v in vals if v <= tolerance)
        per[c.name] = ConstraintValidation(
            name=c.name,
            satisfied_fraction=(sat / n) if n else 0.0,
            mean_violation=(sum(vals) / n) if n else 0.0,
            max_violation=max(vals) if vals else 0.0,
            n=n,
        )
    return ValidationReport(per_constraint=per, tolerance=tolerance)


# --------------------------------------------------------------------------
# Pre-flight (6.1 + backend config checks)
# --------------------------------------------------------------------------

@dataclass
class PreflightReport:
    conflicts: List[ConflictReport] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not any(
            c.severity == "contradiction" for c in self.conflicts
        )

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("preflight failed:\n  " + "\n  ".join(self.errors))

    def __repr__(self) -> str:
        lines = [f"PreflightReport(ok={self.ok})"]
        for c in self.conflicts:
            lines.append(f"  {c!r}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


def preflight(
    assertions: Sequence[Assertion],
    config: Optional[Dict[str, Any]] = None,
    *,
    model: Any = None,
    raise_on_error: bool = False,
) -> PreflightReport:
    """Static pre-flight validation before a training run starts (6.1).

    If ``model`` is supplied, its steerability is checked (Section 7.5): a tree
    ensemble or opaque ``.fit()`` estimator is reported as an error (and raised
    when ``raise_on_error`` is set).
    """
    report = PreflightReport()
    report.conflicts = detect_conflicts(assertions)

    if model is not None:
        from .exceptions import UnsupportedTargetError
        from .target_support import check_steerable

        try:
            check_steerable(model)
        except UnsupportedTargetError as exc:
            report.errors.append(str(exc))
            if raise_on_error:
                raise

    cfg = config or {}
    backend = cfg.get("backend")
    distributed = cfg.get("distributed")

    # JAX distributed: auto cannot be resolved at runtime (4.1b)
    if backend == "jax_native" and distributed == "auto":
        msg = (
            "distributed: auto is invalid under backend: jax_native -- JAX has "
            "no runtime equivalent of torch.distributed.is_initialized(); provide "
            "an explicit axis_name (Section 4.1b)."
        )
        report.errors.append(msg)
        if raise_on_error:
            raise DistributedConfigError(msg)

    # full-model gradient-surgery cost warning on very large models (5.6)
    diags = cfg.get("diagnostics", {}) or {}
    scope = diags.get("gradient_surgery_scope")
    param_count = cfg.get("param_count")
    if scope is not None and param_count is not None:
        from .diagnostics.surgery import full_model_scope_warning

        warn = full_model_scope_warning(scope, int(param_count))
        if warn:
            report.conflicts.append(
                ConflictReport("warning", "full_model_surgery_cost", [], warn)
            )

    # compiled optimizer step + gradient surgery conflict (4.6)
    if diags.get("gradient_surgery") and cfg.get("compiled_optimizer_step"):
        msg = (
            "optimizer step is compiled inside the same region as a "
            "gradient_surgery constraint; gradient surgery needs an eager gap "
            "between backward and step (Section 4.6)."
        )
        report.errors.append(msg)
        if raise_on_error:
            raise CompiledOptimizerStepConflictError(msg)

    return report


# --------------------------------------------------------------------------
# Dry-run feasibility (6.5)
# --------------------------------------------------------------------------

@dataclass
class DryRunResult:
    trending_down: Dict[str, bool]
    start_violation: Dict[str, float]
    end_violation: Dict[str, float]

    @property
    def feasible(self) -> bool:
        return all(self.trending_down.values()) if self.trending_down else False


def dry_run(
    step_fn: Callable[[int], Dict[str, float]],
    constraint_names: Sequence[str],
    *,
    steps: int = 100,
) -> DryRunResult:
    """Run a short trial and report whether violation trends downward (6.5).

    ``step_fn(i)`` performs one training+controller step under aggressive
    weighting and returns the current per-constraint violation dict.
    """
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for i in range(steps):
        v = step_fn(i)
        if i == 0:
            first = dict(v)
        last = dict(v)
    trending = {
        n: (last.get(n, float("inf")) < first.get(n, float("inf")))
        for n in constraint_names
    }
    return DryRunResult(trending_down=trending, start_violation=first, end_violation=last)
