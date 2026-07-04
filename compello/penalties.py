"""Differentiable penalty functions (Section 3.1 table).

Each function converts "this should be true" into a smooth, non-negative
penalty whose value is 0 when the property holds and grows as it is violated.
They are written once against ``compello.math`` so they run identically on any
backend. They return a scalar-like array in the backend's native type; the
controller consumes ``math.to_float(...)`` of it.

Predicates for range/inequality assertions are parsed from either a comparison
string ("> 0", ">= 0.6") or a small callable; a raw Python lambda body cannot
be introspected for dispatch (that is why dispatch lives in the target type,
Section 3.1a) but a lambda CAN be evaluated to derive a threshold when combined
with the target-type-selected penalty shape.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple, Union

from . import math as cmath

Condition = Union[str, Callable[[Any], Any], Tuple[str, float]]

_COND_RE = re.compile(r"^\s*(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_condition(condition: Condition) -> Tuple[str, float]:
    """Normalise a condition into ``(operator, threshold)``.

    Accepts ``"> 0"``, ``(">", 0.0)``, or a callable of one arg that is probed
    with a couple of sentinel values to recover a simple comparison. Only the
    common ``>``/``<``/``>=``/``<=`` forms are supported for probing; anything
    else must be given as a string or tuple.
    """
    if isinstance(condition, tuple) and len(condition) == 2:
        op, thr = condition
        return op, float(thr)
    if isinstance(condition, str):
        m = _COND_RE.match(condition)
        if not m:
            raise ValueError(f"cannot parse condition {condition!r}")
        return m.group(1), float(m.group(2))
    if callable(condition):
        return _probe_predicate(condition)
    raise TypeError(f"unsupported condition type: {type(condition)!r}")


def _probe_predicate(pred: Callable[[Any], Any]) -> Tuple[str, float]:
    """Recover ``(op, threshold)`` from a monotone one-arg predicate.

    Uses a coarse scan to find the boundary where the predicate flips, and the
    direction of the flip to pick the operator. This handles ``lambda y: y > 0``
    and ``lambda p: p > 0.6`` style predicates. Non-monotone predicates raise.
    """
    xs = [i * 0.5 for i in range(-2000, 2001)]  # -1000 .. 1000 step 0.5 (coarse)
    vals = [bool(pred(x)) for x in xs]
    flips = [i for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
    if len(flips) != 1:
        raise ValueError(
            "predicate is not a simple monotone threshold; pass the condition "
            "as a string like '> 0.6' instead of a lambda."
        )
    i = flips[0]
    # direction: True going up -> '>', True going down -> '<'
    op = ">" if (vals[i] and not vals[i - 1]) else "<"
    # bisect between the coarse bracket to pin the boundary precisely
    lo, hi = xs[i - 1], xs[i]
    lo_val = vals[i - 1]
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if bool(pred(mid)) == lo_val:
            lo = mid
        else:
            hi = mid
    thr = round((lo + hi) / 2.0, 6)
    # snap tiny floating dust to a clean value
    if abs(thr - round(thr)) < 1e-6:
        thr = float(round(thr))
    return op, thr


# --------------------------------------------------------------------------
# Penalty shapes
# --------------------------------------------------------------------------

def hinge_range(tensor: Any, op: str, threshold: float) -> Any:
    """Hinge penalty for a range/inequality assertion (Section 3.1).

    For ``x > t`` (or ``>=``) penalise ``relu(t - x)``.
    For ``x < t`` (or ``<=``) penalise ``relu(x - t)``.
    """
    x = cmath.asarray(tensor)
    if op in (">", ">="):
        viol = cmath.relu(threshold - x)
    elif op in ("<", "<="):
        viol = cmath.relu(x - threshold)
    else:
        raise ValueError(f"unsupported operator {op!r} for range penalty")
    return cmath.mean(viol)


def monotonicity(tensor: Any, increasing: bool = True, axis: int = -1) -> Any:
    """Penalise finite differences with the wrong sign along ``axis``.

    For an increasing constraint, negative diffs are violations; penalise
    ``relu(-diff)``. ``tensor`` is expected to be ordered along ``axis`` by the
    feature the monotonicity is declared in.
    """
    d = cmath.diff(cmath.asarray(tensor), axis=axis)
    viol = cmath.relu(-d) if increasing else cmath.relu(d)
    return cmath.mean(viol)


def invariance_l2(output_a: Any, output_b: Any) -> Any:
    """L2 distance between outputs on transformed vs. original input."""
    a = cmath.asarray(output_a)
    b = cmath.asarray(output_b)
    diff = cmath.flatten(a) - cmath.flatten(b)
    return cmath.mean(diff * diff)


def probability_floor(
    logits: Any, index: Any, op: str, threshold: float, mask: Any = None
) -> Any:
    """Penalty on probability shortfall for an indexed logit (Section 3.1a/4.5).

    The predicate is evaluated in probability space: softmax over the last axis,
    then the probability of ``index`` is compared to ``threshold``. Mask-aware:
    if ``mask`` is provided the per-position violation is multiplied by it so
    only real target positions contribute (Section 4.5 packed-sequence safety).
    """
    probs = cmath.softmax(cmath.asarray(logits), axis=-1)
    p = _index_last_axis(probs, index)
    if op in (">", ">="):
        viol = cmath.relu(threshold - p)
    elif op in ("<", "<="):
        viol = cmath.relu(p - threshold)
    else:
        raise ValueError(f"unsupported operator {op!r} for probability floor")
    if mask is not None:
        viol = viol * cmath.asarray(mask)
        denom = cmath.sum(cmath.asarray(mask))
        total = cmath.sum(viol)
        return total / (cmath.to_float(denom) + 1e-12)
    return cmath.mean(viol)


def parity(rates_group_a: Any, rates_group_b: Any) -> Any:
    """Penalty on cross-group rate difference (fairness-style)."""
    a = cmath.mean(cmath.asarray(rates_group_a))
    b = cmath.mean(cmath.asarray(rates_group_b))
    return cmath.abs(a - b)


def lipschitz(input_output_grad_norm: Any, bound: float) -> Any:
    """Penalty on input-output gradient norm exceeding ``bound``.

    Expects the caller (backend adapter) to supply the already-computed
    per-sample gradient norms; the portable penalty only enforces the ceiling.
    """
    g = cmath.asarray(input_output_grad_norm)
    return cmath.mean(cmath.relu(g - bound))


def consistency(rep_a: Any, rep_b: Any) -> Any:
    """Penalty on disagreement between paired representations (cross-view)."""
    return invariance_l2(rep_a, rep_b)


def sigmoid_relaxation(margin: Any, alpha: float) -> Any:
    """Smooth relaxation of a hard threshold (Section 4.3).

    Returns ``1 - sigmoid(alpha * margin)`` so that satisfying the threshold
    (positive margin) drives the penalty toward 0. ``alpha`` (sharpness) is
    tuned adaptively with hysteresis by the controller.
    """
    m = cmath.asarray(margin)
    return cmath.mean(1.0 - cmath.sigmoid(alpha * m))


def _index_last_axis(arr: Any, index: Any) -> Any:
    """Index the last axis of ``arr`` by ``index`` in a backend-agnostic way.

    Works for numpy arrays and any backend tensor supporting ``[..., index]``.
    """
    try:
        return arr[..., index]
    except Exception:
        # Fall back to plain indexing for 1-D probability vectors.
        return arr[index]
