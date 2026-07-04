"""Batched distributed-sync discipline (Section 4.1 / 4.1b).

The efficiency property that matters at scale is *one collective per step*: all
active constraints' local violation scalars are packed into a single array,
reduced once across replicas, then unpacked -- never one collective per
constraint. That discipline is backend-agnostic; only the collective call
itself differs (``all_reduce`` / ``strategy.reduce`` / ``lax.pmean``).

``batched_sync`` is that portable discipline, tested with a numpy reduce. Each
adapter passes a ``reduce_fn`` that wraps its framework's collective, so the
"exactly one synchronization pass per macro-step" guarantee lives in one tested
place rather than being re-implemented (and re-verified) three times.
"""

from __future__ import annotations

from typing import Callable, Dict, List


def batched_sync(
    local_violations: Dict[str, float],
    reduce_fn: Callable[[List[float]], List[float]],
) -> Dict[str, float]:
    """Stack -> single reduce -> unpack.

    ``reduce_fn`` receives the list of per-constraint local violations in a
    stable name order and must return the globally-reduced list in the same
    order. It is invoked exactly once regardless of the number of constraints.
    """
    if not local_violations:
        return {}
    names = list(local_violations)
    stacked = [local_violations[n] for n in names]
    reduced = reduce_fn(stacked)
    if len(reduced) != len(names):
        raise ValueError(
            "reduce_fn must return one value per constraint "
            f"({len(names)} expected, got {len(reduced)})"
        )
    return {n: float(reduced[i]) for i, n in enumerate(names)}
