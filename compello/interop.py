"""Cooper interoperability export (Section 6.6).

A declared Compello constraint set can be handed off to Cooper for a production
run. The handoff is a *snapshot of current multiplier values*, not a full state
transplant: Cooper has no slot for Compello's EMA buffers, dual-rate spike
detection + rolling baseline, adaptive sharpness + hysteresis, or the
momentum-aware grace window, so those are dropped. This module builds an
explicit, inspectable description of exactly what transfers and what is lost, so
the lossy nature of the conversion is visible rather than implied.

Constructing the actual ``cooper`` objects requires ``cooper`` (and ``torch``);
that happens in ``compello.backends.torch`` when the library is present. The
description below is framework-independent and always available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .assertions import PROBABILITY_FLOOR, PROXY, RANGE, Assertion

# state fields that have no Cooper equivalent and are therefore dropped (6.6)
DROPPED_STATE = [
    "slow_ema_buffer",
    "fast_ema_buffer",
    "dual_rate_spike_detection",
    "rolling_baseline",
    "adaptive_sharpness",
    "sharpness_hysteresis_arm",
    "momentum_aware_grace_window",
]


@dataclass
class CooperConstraintExport:
    name: str
    multiplier: float          # transfers cleanly
    kind: str
    condition: Optional[str]
    transferred: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)


@dataclass
class CooperExport:
    constraints: List[CooperConstraintExport]
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["CooperExport (snapshot of multiplier values only):"]
        for c in self.constraints:
            cond = f" {c.condition}" if c.condition else ""
            lines.append(f"  {c.name} [{c.kind}{cond}] -> multiplier={c.multiplier:g}")
            if c.dropped:
                lines.append(f"    dropped (no Cooper equivalent): {', '.join(c.dropped)}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def build_cooper_export(
    assertions: Sequence[Assertion], *, controller: Any = None
) -> CooperExport:
    """Describe the (lossy) Cooper handoff for ``assertions`` (Section 6.6)."""
    weights: Dict[str, float] = {}
    if controller is not None and hasattr(controller, "weights"):
        weights = controller.weights

    exports: List[CooperConstraintExport] = []
    for a in assertions:
        cond = f"{a.op} {a.threshold}" if a.op is not None else None
        dropped = list(DROPPED_STATE)
        if a.kind != PROXY:
            # sharpness state only meaningful for proxy constraints
            dropped = [d for d in dropped if "sharpness" not in d]
        exports.append(
            CooperConstraintExport(
                name=a.name,
                multiplier=float(weights.get(a.name, a.initial_weight)),
                kind=a.kind,
                condition=cond,
                transferred=["multiplier", "dual_optimizer_state"],
                dropped=dropped,
            )
        )

    notes = [
        "Only current multiplier values transfer; training continues under "
        "Cooper's own (un-smoothed) dual-ascent dynamics from that point.",
    ]
    if any(a.kind == PROXY for a in assertions):
        notes.append(
            "Proxy constraints carry only their current fixed sharpness value; "
            "further adaptive sharpness tuning stops after export."
        )
    return CooperExport(constraints=exports, notes=notes)


def export_to_cooper_objects(assertions: Sequence[Assertion], *, controller: Any = None):
    """Construct real Cooper multiplier objects when ``cooper`` is installed (6.6).

    Returns ``(CooperExport, multipliers)`` where ``multipliers`` maps each
    constraint name to a ``cooper`` multiplier initialised to the constraint's
    current Compello weight. If ``cooper`` (and ``torch``) are not installed,
    ``multipliers`` is ``None`` and only the descriptive, lossy-conversion
    ``CooperExport`` is returned -- the dropped-state warnings still apply and are
    attached to the export's ``notes``.
    """
    export = build_cooper_export(assertions, controller=controller)
    try:
        import cooper  # noqa: F401
        import torch
    except Exception:
        export.notes.append(
            "cooper/torch not installed: returning the multiplier-snapshot "
            "description only; no live Cooper objects were constructed."
        )
        return export, None

    multipliers: Dict[str, Any] = {}
    for c in export.constraints:  # pragma: no cover - requires cooper+torch
        init = torch.tensor(float(c.multiplier))
        made = None
        for factory in ("multipliers", "formulation"):
            mod = getattr(cooper, factory, None)
            cls = getattr(mod, "DenseMultiplier", None) if mod else None
            if cls is not None:
                try:
                    made = cls(init=init)
                    break
                except Exception:
                    continue
        multipliers[c.name] = made if made is not None else init
    export.notes.append(
        "Constructed Cooper multipliers from current weights; EMA/spike/"
        "sharpness/grace state was NOT transferred (no Cooper equivalent)."
    )
    return export, multipliers
