"""Pre-flight, static constraint-conflict detection (Section 6.1).

Before any compute is spent, the declared constraint set is checked for direct
logical contradictions, structural tension, and redundancy. This runs on the
declared ``Assertion`` objects (grouped by ``target_id``) -- no training step is
executed. It surfaces high-risk pairs so they can be fixed before a run starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence

from ..assertions import (
    INVARIANCE,
    MONOTONICITY,
    PROBABILITY_FLOOR,
    RANGE,
    Assertion,
)

CONTRADICTION = "contradiction"
TENSION = "tension"
REDUNDANCY = "redundancy"


@dataclass
class ConflictReport:
    severity: str          # CONTRADICTION | TENSION | REDUNDANCY
    kind: str              # short machine tag
    assertions: List[str]  # names of involved assertions
    message: str

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.message}"


def detect_conflicts(assertions: Sequence[Assertion]) -> List[ConflictReport]:
    reports: List[ConflictReport] = []
    for a, b in combinations(assertions, 2):
        if a.target_id is None or b.target_id is None or a.target_id != b.target_id:
            continue
        reports.extend(_pair_conflict(a, b))
    return reports


def _pair_conflict(a: Assertion, b: Assertion) -> List[ConflictReport]:
    out: List[ConflictReport] = []

    # 1) contradictory range/probability conditions on the same target
    if a.kind in (RANGE, PROBABILITY_FLOOR) and b.kind == a.kind:
        if _range_contradiction(a, b):
            out.append(ConflictReport(
                CONTRADICTION, "range_contradiction", [a.name, b.name],
                f"{a.name} ({a.op} {a.threshold}) and {b.name} ({b.op} {b.threshold}) "
                f"on the same target cannot both hold.",
            ))
        elif a.op == b.op and a.threshold == b.threshold:
            out.append(ConflictReport(
                REDUNDANCY, "duplicate", [a.name, b.name],
                f"{a.name} and {b.name} declare the same condition on the same target.",
            ))

    # 2) structural tension: invariance vs monotonicity on the same target
    kinds = {a.kind, b.kind}
    if kinds == {INVARIANCE, MONOTONICITY}:
        out.append(ConflictReport(
            TENSION, "invariance_vs_monotonicity", [a.name, b.name],
            f"{a.name} and {b.name} imply opposite behaviour on the same target: "
            f"invariance wants the output unchanged, monotonicity wants it to move.",
        ))
    return out


def _range_contradiction(a: Assertion, b: Assertion) -> bool:
    """True if two one-sided conditions have an empty satisfiable intersection."""
    if a.threshold is None or b.threshold is None:
        return False
    # normalise to (lower_bound, upper_bound) half-lines
    def bounds(op, t):
        if op in (">", ">="):
            return (t, float("inf"))
        if op in ("<", "<="):
            return (float("-inf"), t)
        return (float("-inf"), float("inf"))

    la, ua = bounds(a.op, a.threshold)
    lb, ub = bounds(b.op, b.threshold)
    lo = max(la, lb)
    hi = min(ua, ub)
    return lo > hi
