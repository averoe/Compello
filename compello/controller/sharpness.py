"""Adaptive proxy sharpness with hysteresis (Section 4.3).

Non-differentiable targets are relaxed with a scaled sigmoid ``sigma(alpha*x)``.
If alpha is too high the gradient collapses to zero the moment a batch leaves
the narrow active window -- a dead end, since a zero gradient never signals "get
closer." Compello monitors the running proxy gradient norm and lowers alpha to
widen the active window when it vanishes, then re-sharpens once signal returns.

Two distinct thresholds (hysteresis) prevent chatter at the boundary:
  scale-down when ``grad_norm < g_floor``
  scale-up   when ``grad_norm > g_floor * sharpness_hysteresis``
The default multiplier 1.5 separates the two triggers by more than three std of
typical (30-40%) step-to-step gradient-norm noise.
"""

from __future__ import annotations

from .state import ConstraintState

ALPHA_MIN = 0.1
ALPHA_MAX = 100.0
SCALE_DOWN_FACTOR = 0.5   # exponential widening
SCALE_UP_FACTOR = 1.25    # progressive re-sharpening


def update_sharpness(
    state: ConstraintState,
    proxy_grad_norm: float,
    metric_satisfied: bool,
    *,
    g_floor: float,
    sharpness_hysteresis: float = 1.5,
    sharpness_patience: int = 0,
) -> float:
    """Adjust ``state.alpha`` and return the new value.

    ``metric_satisfied`` is whether the *true* underlying metric already meets
    target; a vanished gradient only warrants widening when the metric still
    fails (otherwise a small gradient is fine -- we are already correct).

    Dead-band lockup fix (1.5): if the gradient norm stabilises *inside* the
    dead-band (between the scale-down trigger ``g_floor`` and the scale-up re-arm
    band ``g_floor * sharpness_hysteresis``) while disarmed, the controller can
    never re-arm and adaptation freezes. When ``sharpness_patience > 0``, after
    the norm has sat in the dead-band for that many steps the *effective* re-arm
    threshold decays toward ``g_floor`` so the controller can re-arm and resume
    adaptation instead of locking up permanently.
    """
    down_trigger = proxy_grad_norm < g_floor

    effective_hysteresis = sharpness_hysteresis
    if not state.sharpness_armed and sharpness_patience > 0:
        in_deadband = g_floor <= proxy_grad_norm <= g_floor * sharpness_hysteresis
        if in_deadband:
            state.deadband_steps += 1
        else:
            state.deadband_steps = 0
        if state.deadband_steps >= sharpness_patience:
            # decay the re-arm band toward the floor proportional to how long
            # we've been stuck, so a parked norm eventually re-arms.
            over = state.deadband_steps - sharpness_patience
            decay = min(1.0, over / max(1, sharpness_patience))
            effective_hysteresis = (
                sharpness_hysteresis - (sharpness_hysteresis - 1.0) * decay
            )

    up_trigger = proxy_grad_norm > g_floor * effective_hysteresis

    if state.sharpness_armed and down_trigger and not metric_satisfied:
        # vanishing zone: widen the window to restore gradient signal
        state.alpha = max(ALPHA_MIN, state.alpha * SCALE_DOWN_FACTOR)
        state.sharpness_armed = False  # disarm until above the re-arm band
        state.deadband_steps = 0
    elif not state.sharpness_armed and up_trigger:
        # back in the effective range (possibly via patience decay): re-sharpen
        state.alpha = min(ALPHA_MAX, state.alpha * SCALE_UP_FACTOR)
        state.sharpness_armed = True
        state.deadband_steps = 0
    return state.alpha
