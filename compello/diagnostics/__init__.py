"""Compello diagnostics (Sections 5.1a, 5.2, 5.3, 6.1)."""

from .coldstart import ColdStartMonitor, ColdStartState
from .runner import DiagnosticsRunner
from .conflict import (
    CONTRADICTION,
    REDUNDANCY,
    TENSION,
    ConflictReport,
    detect_conflicts,
)
from .regression import FitResult, RollingRegressor, steps_from_rate
from .surgery import (
    ScopedSurgeryResult,
    SurgeryResult,
    apply_gradient_surgery,
    cosine_similarity,
    full_model_scope_warning,
    project_out_conflict,
    scoped_gradient_surgery,
    select_in_scope,
)

__all__ = [
    "apply_gradient_surgery",
    "scoped_gradient_surgery",
    "select_in_scope",
    "full_model_scope_warning",
    "project_out_conflict",
    "cosine_similarity",
    "SurgeryResult",
    "ScopedSurgeryResult",
    "RollingRegressor",
    "FitResult",
    "steps_from_rate",
    "detect_conflicts",
    "ConflictReport",
    "CONTRADICTION",
    "TENSION",
    "REDUNDANCY",
    "ColdStartMonitor",
    "ColdStartState",
    "DiagnosticsRunner",
]
