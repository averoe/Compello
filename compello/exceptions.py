"""Named exceptions for Compello.

These are referenced throughout the spec (Sections 3.1a, 4.1b, 4.6, 6, 7.5) and
are surfaced primarily by ``trainlint``'s pre-flight pass so that
misconfigurations become clear, named errors before a training run starts
rather than confusing stack traces deep inside the controller.
"""

from __future__ import annotations


class CompelloError(Exception):
    """Base class for all Compello errors."""


class AmbiguousAssertionError(CompelloError):
    """Raised when ``expect()`` cannot resolve which penalty to apply.

    Happens when an assertion targets a raw value that has no typed wrapper
    (Section 3.0/3.1a) and no explicit ``assertion_type=`` keyword was given.
    """


class BackendNotAvailableError(CompelloError):
    """Raised when an operation requires a backend that is not installed/registered."""


class UnsupportedTargetError(CompelloError):
    """Raised when a target is not a differentiable, controllable model.

    Covers the Section 7.5 boundary: tree ensembles (RandomForest, XGBoost,
    LightGBM) and opaque ``.fit()`` estimators such as scikit-learn's
    ``SGDClassifier`` cannot be steered because there is no per-step gradient
    interface to intervene on.
    """


class CompiledOptimizerStepConflictError(CompelloError):
    """Raised when the optimizer step is compiled inside the same region as a
    gradient-surgery constraint (Section 4.6).

    Gradient Surgery needs a post-backward, pre-step eager-mode gap that does
    not exist when ``optimizer.step()`` is enclosed in the compiled graph.
    """


class DistributedConfigError(CompelloError):
    """Raised for invalid distributed configuration.

    Notably the JAX ``distributed: auto`` case (Section 4.1b): JAX has no
    runtime equivalent of ``torch.distributed.is_initialized()``, so an
    explicit ``axis_name`` is required.
    """


class InfeasibleConstraintError(CompelloError):
    """Raised (or reported) when a constraint cannot be satisfied.

    Either detected by the dry-run feasibility check (6.5) or after the
    controller's weight ceiling is hit and violation still will not fall (4.4).
    """


class ConstraintConflictError(CompelloError):
    """Raised by pre-flight conflict detection (6.1) for directly contradictory
    constraint sets."""
