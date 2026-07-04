"""Compello controller package (Section 3.2 / 4.2-4.4)."""

from .ceiling import apply_ceiling
from .controller import (
    ADAPTIVE_PID,
    DUAL_ASCENT,
    FIXED,
    LINEAR_RAMP,
    ConstraintStep,
    Controller,
    ControllerConfig,
    StepResult,
)
from .ema import ema_update, update_ema_layer
from .pid import PIDGains, pid_step
from .sharpness import update_sharpness
from .state import ConstraintState

__all__ = [
    "Controller",
    "ControllerConfig",
    "StepResult",
    "ConstraintStep",
    "ConstraintState",
    "PIDGains",
    "pid_step",
    "ema_update",
    "update_ema_layer",
    "update_sharpness",
    "apply_ceiling",
    "FIXED",
    "LINEAR_RAMP",
    "ADAPTIVE_PID",
    "DUAL_ASCENT",
]
