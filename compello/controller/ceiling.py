"""Dynamic weight ceiling -- primary-loss-eradication guard (Section 4.4).

An unmanaged multiplier ramps without bound until the constraint gradient
dwarfs the task gradient and the optimiser abandons the real objective for a
degenerate shortcut that trivially satisfies the constraint. Compello caps each
constraint weight at ``weight_ceiling`` (relative to the primary task weight)
and, if a constraint would need to exceed the ceiling to converge, locks the
weight there and raises a non-convergence diagnostic rather than sacrificing the
primary objective.
"""

from __future__ import annotations

from .state import ConstraintState


def apply_ceiling(
    state: ConstraintState,
    proposed_weight: float,
    *,
    weight_ceiling: float,
    still_violated: bool,
) -> float:
    """Clamp ``proposed_weight`` to the ceiling and set the lock/infeasible flag.

    Returns the effective weight. If the ceiling is hit while the constraint is
    still violated, ``state.ceiling_locked`` is set -- surfaced by diagnostics
    (Section 6.3) as "possibly architecturally infeasible at current capacity."
    """
    if proposed_weight >= weight_ceiling:
        effective = weight_ceiling
        state.ceiling_locked = bool(still_violated)
    else:
        effective = max(0.0, proposed_weight)
        state.ceiling_locked = False
    return effective
