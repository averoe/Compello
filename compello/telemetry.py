"""Live, plain-English telemetry and insight rendering (Sections 5.1a, 5.5).

Renders the controller's decisions into human-readable lines. Numeric
projections carry a confidence qualifier when the underlying rolling-fit R^2 is
below 0.5 and are suppressed below 0.2 (Section 5.1a), so telemetry never
presents a random-walk fit as an authoritative number.

``telemetry`` verbosity: ``compact`` | ``verbose`` | ``silent``.
"""

from __future__ import annotations

from typing import Optional

from .controller.controller import StepResult

COMPACT = "compact"
VERBOSE = "verbose"
SILENT = "silent"


def render_step(
    result: StepResult,
    *,
    verbosity: str = COMPACT,
    total_steps: Optional[int] = None,
    loss: Optional[float] = None,
) -> str:
    if verbosity == SILENT:
        return ""
    active = [c for c in result.per_constraint.values() if not c.satisfied]
    n_total = len(result.per_constraint)
    n_active = len(active)
    header_pct = ""
    if total_steps:
        header_pct = f" | {100 * result.step / total_steps:.0f}%"
    loss_str = f" | loss {loss:.4g}" if loss is not None else ""
    header = (
        f"Step {result.step}{header_pct}{loss_str} | "
        f"{n_total - n_active}/{n_total} constraints stable"
    )

    if verbosity == COMPACT:
        lines = [header]
        for c in active:
            lines.append(
                f"  '{c.name}' active - violation {c.raw_violation:.4g}, "
                f"weight {'+' if c.weight_delta >= 0 else ''}{c.weight_delta:.3g}"
            )
        for c in result.per_constraint.values():
            if c.satisfied and c.weight_delta < 0:
                lines.append(
                    f"  '{c.name}' stable - relaxing weight "
                    f"{100 * c.weight_delta / max(1e-9, c.weight - c.weight_delta):.0f}%, "
                    f"returning capacity to task loss."
                )
        return "\n".join(lines)

    # VERBOSE
    lines = [header]
    for c in result.per_constraint.values():
        lines.append(
            f"  '{c.name}': raw={c.raw_violation:.4g} smoothed={c.smoothed_violation:.4g} "
            f"weight={c.weight:.4g} (d={c.weight_delta:+.3g}) alpha={c.alpha:.3g} "
            f"{'SATISFIED' if c.satisfied else 'ACTIVE'}"
            f"{' [CEILING-LOCKED]' if c.ceiling_locked else ''}"
        )
    return "\n".join(lines)


def render_recovery_estimate(
    constraint_name: str,
    steps: Optional[float],
    confidence: str,
    *,
    r_squared: Optional[float] = None,
) -> str:
    """Render a recovery-time insight with the 5.1a confidence gating."""
    r2 = f", R^2={r_squared:.2f}" if r_squared is not None else ""
    if confidence == "suppressed" or steps is None:
        return (
            f"[Compello Insight] '{constraint_name}': recovery trend unclear from "
            f"recent steps{r2} - projection suppressed."
        )
    qualifier = "" if confidence == "reliable" else " (low confidence"
    tail = "" if confidence == "reliable" else f"{r2})"
    if confidence == "reliable":
        return (
            f"[Compello Insight] '{constraint_name}': estimated recovery ~{steps:.0f} "
            f"steps (rolling-fit estimate{r2})."
        )
    return (
        f"[Compello Insight] '{constraint_name}': estimated recovery ~{steps:.0f} "
        f"steps{qualifier}{tail}."
    )


def render_gradient_conflict(constraint_name: str, cosine: float, projected: bool, layer: Optional[str] = None) -> str:
    loc = f" concentrated at {layer}" if layer else ""
    action = (
        "Gradient surgery applied to remove the conflicting component."
        if projected else "No projection needed."
    )
    return (
        f"[Compello Insight] Gradient conflict between task loss and "
        f"'{constraint_name}' (cosine {cosine:.2f}){loc}. {action}"
    )
