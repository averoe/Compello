"""Compello -- a constraint-driven autotraining framework.

Public API (framework-independent core). Deep-learning backend adapters
(torch / tensorflow / jax) live under ``compello.backends`` and are imported
only when their framework is present. The core here runs against the numpy
reference backend so its math, controller, and diagnostics are fully usable and
testable without any deep-learning framework installed.
"""

from __future__ import annotations

from . import math
from .assertions import (
    Assertion,
    expect,
    register_assertion_type,
    registered_assertion_types,
)
from .controller import (
    ADAPTIVE_PID,
    DUAL_ASCENT,
    FIXED,
    LINEAR_RAMP,
    Controller,
    ControllerConfig,
)
from .diagnostics import (
    ColdStartMonitor,
    ConflictReport,
    DiagnosticsRunner,
    RollingRegressor,
    ScopedSurgeryResult,
    apply_gradient_surgery,
    detect_conflicts,
    full_model_scope_warning,
    scoped_gradient_surgery,
    select_in_scope,
)
from .distillation import DistillationBridge, distillation_bridge
from .serialization import load_controller, save_controller
from .surrogates import (
    FTTransformerReference,
    NodeReference,
    build_ft_transformer,
    build_node,
)
from .insights import InsightEngine, StepInsights
from .relaxations import (
    register_builtin_relaxations,
    soft_f1_penalty,
    soft_iou_penalty,
    soft_rank_penalty,
    spectral_gate_penalty,
)
from .preflight_render import render_preflight_shield
from .report_style import Style
from .reports import (
    SensitivityProfiler,
    non_convergence_report,
    render_capacity_report,
)
from .exceptions import (
    AmbiguousAssertionError,
    BackendNotAvailableError,
    CompelloError,
    CompiledOptimizerStepConflictError,
    ConstraintConflictError,
    DistributedConfigError,
    InfeasibleConstraintError,
    UnsupportedTargetError,
)
from .config import CompelloConfig, ConstraintSpec, load_config
from .proxy import ModelProxy, TensorProxy, unwrap, wrap
from .target_support import check_steerable, is_steerable
from .targets import LogitTarget, ModelTarget, OutputTarget
from .validation import (
    DryRunResult,
    PreflightReport,
    ValidationReport,
    dry_run,
    preflight,
    validate,
)

__version__ = "0.1.0"

# Register the modality proxy relaxations (5.1) as custom assertion types so
# they're usable via assertion_type="soft_iou" etc. out of the box.
register_builtin_relaxations()

__all__ = [
    "__version__",
    "math",
    # declaration
    "wrap",
    "unwrap",
    "expect",
    "Assertion",
    "ModelProxy",
    "TensorProxy",
    "OutputTarget",
    "LogitTarget",
    "ModelTarget",
    # control
    "Controller",
    "ControllerConfig",
    "FIXED",
    "LINEAR_RAMP",
    "ADAPTIVE_PID",
    "DUAL_ASCENT",
    # diagnostics
    "apply_gradient_surgery",
    "detect_conflicts",
    "ConflictReport",
    "RollingRegressor",
    "ColdStartMonitor",
    # validation / preflight
    "validate",
    "preflight",
    "dry_run",
    "ValidationReport",
    "PreflightReport",
    "DryRunResult",
    # config
    "load_config",
    "CompelloConfig",
    "ConstraintSpec",
    "export_to_cooper",
    # target support (7.5)
    "check_steerable",
    "is_steerable",
    # custom assertion types (3.1)
    "register_assertion_type",
    "registered_assertion_types",
    # modality relaxations (5.1)
    "soft_iou_penalty",
    "soft_f1_penalty",
    "spectral_gate_penalty",
    "soft_rank_penalty",
    # insight engine + reports (5.1a, 5.4, 5.5, 6.3)
    "InsightEngine",
    "StepInsights",
    "DiagnosticsRunner",
    "SensitivityProfiler",
    "non_convergence_report",
    # rich terminal renderers (5.4, 5.5, 8)
    "render_preflight_shield",
    "render_capacity_report",
    "Style",
    # classical-ML distillation bridge + surrogates (7.5)
    "distillation_bridge",
    "DistillationBridge",
    "build_node",
    "build_ft_transformer",
    "NodeReference",
    "FTTransformerReference",
    # layer-scoped surgery (5.6)
    "scoped_gradient_surgery",
    "select_in_scope",
    "full_model_scope_warning",
    "ScopedSurgeryResult",
    # checkpoint serialization (6.2)
    "save_controller",
    "load_controller",
    # exceptions
    "CompelloError",
    "AmbiguousAssertionError",
    "BackendNotAvailableError",
    "UnsupportedTargetError",
    "CompiledOptimizerStepConflictError",
    "DistributedConfigError",
    "InfeasibleConstraintError",
    "ConstraintConflictError",
]


def export_to_cooper(assertions, *, controller=None):
    """Export a constraint set to Cooper's native objects (Section 6.6).

    Only a *snapshot of current multiplier values* transfers. Compello-specific
    state (EMA buffers, dual-rate spike detection + rolling baseline, adaptive
    sharpness + hysteresis, momentum-aware grace window) has no Cooper
    equivalent and is intentionally dropped -- see Section 6.6. This function
    returns a plain, serialisable description of what would be handed off so the
    lossy nature of the conversion is explicit and inspectable; the actual
    ``cooper`` object construction requires ``cooper`` (and thus ``torch``) to be
    installed and is performed by ``compello.backends.torch`` when present.
    """
    from .interop import build_cooper_export

    return build_cooper_export(assertions, controller=controller)


def export_to_cooper_objects(assertions, *, controller=None):
    """Construct real Cooper multiplier objects when cooper+torch are installed,
    else return the lossy description only (Section 6.6)."""
    from .interop import export_to_cooper_objects as _impl

    return _impl(assertions, controller=controller)
