"""PID update on the smoothed violation signal (Sections 3.2, 4.2).

Framework-neutral scalar recurrence: the proportional term reacts to current
(smoothed) violation, the integral catches slow persistent violation, and the
derivative dampens oscillation. The derivative acts on the *smoothed* signal,
never the raw one, which is what prevents the derivative kick (4.2).
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import ConstraintState


@dataclass
class PIDGains:
    kp: float = 1.0
    ki: float = 0.01
    kd: float = 0.1
    integral_clip: float = 1e6  # anti-windup bound on the integral accumulator


def pid_step(state: ConstraintState, smoothed_violation: float, gains: PIDGains) -> float:
    """Return the PID control output (the raw weight adjustment signal).

    ``error`` is the smoothed violation itself (target violation is 0). Positive
    error means the constraint is violated and weight pressure should rise.
    """
    error = smoothed_violation

    # integral with anti-windup clamp
    state.integral += error
    if state.integral > gains.integral_clip:
        state.integral = gains.integral_clip
    elif state.integral < -gains.integral_clip:
        state.integral = -gains.integral_clip

    derivative = error - state.prev_error
    state.prev_error = error

    return gains.kp * error + gains.ki * state.integral + gains.kd * derivative
