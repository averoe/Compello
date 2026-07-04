"""Dual-rate EMA smoothing and the sustained-spike detector (Section 4.2).

The raw violation signal is never fed to the PID controller directly -- it is
noisy step-to-step because of stochastic mini-batch sampling, and a single
outlier batch would cause a derivative kick. A slow EMA smooths the signal that
reaches the PID; a faster EMA acts only as a *detector* of sustained shifts so
the controller can shorten its effective lag when a real drift begins.
"""

from __future__ import annotations

from .state import ConstraintState


def ema_update(prev: float, raw: float, beta: float) -> float:
    """Smoothed_t = beta * Smoothed_{t-1} + (1 - beta) * Raw_t."""
    if prev is None:
        return raw
    return beta * prev + (1.0 - beta) * raw


def update_ema_layer(
    state: ConstraintState,
    raw_violation: float,
    *,
    ema_decay: float,
    ema_fast_decay: float,
    ema_override_steps: int,
) -> float:
    """Update slow/fast EMAs and run the spike detector.

    Returns the smoothed value to feed the PID controller. Sets
    ``state.override_remaining`` when a sustained, above-baseline spike is
    confirmed for ``ema_override_steps`` consecutive steps; during an active
    override the effective slow-EMA decay is temporarily moved toward the fast
    decay so the controller catches up faster without ever exposing the D-term
    to single-step noise.
    """
    state.last_raw_violation = raw_violation

    # fast EMA (detector only)
    state.fast = ema_update(state.fast, raw_violation, ema_fast_decay)
    state.fast_history.append(state.fast)

    # detect elevation vs. trailing baseline, excluding the candidate window
    stats = state.baseline_stats(exclude_recent=ema_override_steps)
    elevated = False
    if stats is not None:
        mean, std = stats
        # "elevated" = fast exceeds baseline by more than one trailing std,
        # so sensitivity scales with how jittery the signal normally is (4.2).
        elevated = state.fast > (mean + std)

    if elevated:
        state.consecutive_elevated += 1
    else:
        state.consecutive_elevated = 0

    if state.consecutive_elevated >= ema_override_steps:
        # bound the override to ~override_steps so lag is bounded, not the
        # slow EMA's full settling time.
        state.override_remaining = ema_override_steps

    # effective decay: move toward fast decay while an override is active
    if state.override_remaining > 0:
        effective_decay = ema_fast_decay
        state.override_remaining -= 1
    else:
        effective_decay = ema_decay

    state.smoothed = ema_update(state.smoothed, raw_violation, effective_decay)
    return state.smoothed
