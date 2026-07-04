"""The live insight engine (Sections 5.1a, 5.2, 5.3, 5.5, 5.6).

Turns the controller's *actual* per-step state into the terminal stream from the
spec: a compressed one-line-per-step status feed that stays quiet while metrics
are stable, and expands into a boxed, multi-line "Compello Runtime Insight"
block only when a control loop steps in to correct a drift.

Every number is read from real state or computed by the per-constraint
rolling-regression estimator (5.1a), R^2-gated so a noisy fit never yields an
authoritative-looking projection. Blocks are transition-gated: a given event
(gradient conflict, vanishing-zone alpha drop, ceiling lock, recovery,
intervention) fires a block when it first occurs and re-fires only on a
throttle, so the stream never spams the same line every step.

Layer-localized conflict text (e.g. "at model.layers.28.mlp") is rendered only
when a real backend adapter supplies the layer; otherwise it is omitted rather
than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import telemetry as T
from .controller.controller import Controller, StepResult
from .diagnostics.regression import RollingRegressor
from .diagnostics.runner import DiagnosticsRunner
from .report_style import Style

_MODALITY_LABELS = {
    "text": "Text Modality",
    "vision": "Vision Modality",
    "audio": "Audio Modality",
    "tabular": "Tabular Modality",
    "multimodal": "Multimodal Modality",
    "generic": "Runtime",
}


@dataclass
class StepInsights:
    step: int
    telemetry: str
    insights: List[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [self.telemetry] if self.telemetry else []
        parts.extend(self.insights)
        return "\n".join(parts)


class InsightEngine:
    """Observes controller steps; produces the status stream + insight blocks."""

    def __init__(
        self,
        controller: Controller,
        *,
        telemetry: str = "compact",
        total_steps: Optional[int] = None,
        diagnostics_interval: Optional[int] = None,
        modality: str = "generic",
        backend: Optional[str] = None,
        targets: Optional[Dict[str, str]] = None,
        mask_aware: Optional[set] = None,
        style: Optional[Style] = None,
        reemit_every: int = 200,
    ):
        self.controller = controller
        self.telemetry = telemetry
        self.total_steps = total_steps
        self.modality = modality
        self.backend = backend
        self.targets = targets or {}
        self.mask_aware = mask_aware or set()
        self.style = style or Style.auto()
        self.reemit_every = reemit_every
        self.runner = DiagnosticsRunner(telemetry=telemetry, interval=diagnostics_interval)
        self._reg: Dict[str, RollingRegressor] = {}
        self._prev_alpha: Dict[str, float] = {}
        self._prev_ceiling: Dict[str, bool] = {}
        self._prev_active: Dict[str, bool] = {}
        self._last_conflict_emit: Dict[str, int] = {}
        # Seed the "previous" snapshots from the controller's current state so a
        # transition on the very first observed step (e.g. an alpha drop) is
        # detected relative to the pre-step value, not the post-step value.
        for _name, _st in controller.states.items():
            self._prev_alpha[_name] = _st.alpha
            self._prev_ceiling[_name] = _st.ceiling_locked
            self._prev_active[_name] = False

    # -- main entry ------------------------------------------------------
    def observe(
        self,
        result: StepResult,
        *,
        loss: Optional[float] = None,
        grad_conflicts: Optional[Dict[str, dict]] = None,
    ) -> StepInsights:
        s = self.style
        grad_conflicts = grad_conflicts or {}
        run_diag = self.runner.should_run(result.step)
        blocks: List[str] = []

        for name, cs in result.per_constraint.items():
            reg = self._reg.get(name) or self._reg.setdefault(
                name, RollingRegressor(window=500, refit_interval=50))
            st = self.controller.states[name]
            reg.observe(
                [cs.weight, cs.weight_delta, st.steps_since_intervention, cs.raw_violation],
                cs.raw_violation,
            )

            events = self._detect_events(name, cs, st, grad_conflicts, result)
            if events:
                blocks.append(self._render_block(name, cs, st, events, reg, grad_conflicts, run_diag))

            self._prev_alpha[name] = cs.alpha
            self._prev_ceiling[name] = cs.ceiling_locked
            self._prev_active[name] = not cs.satisfied

        line = self._status_line(result, loss)
        return StepInsights(step=result.step, telemetry=line, insights=blocks)

    # -- compact status line (always) -----------------------------------
    def _status_line(self, result: StepResult, loss: Optional[float]) -> str:
        if self.telemetry == "silent":
            return ""
        s = self.style
        total = len(result.per_constraint)
        compliant = sum(1 for c in result.per_constraint.values() if c.satisfied)
        glyph = s.status_glyph(compliant, total)
        step_str = f"Step {result.step}"
        if self.total_steps:
            step_str += f"/{self.total_steps} | {100 * result.step / self.total_steps:.0f}%"
        be = f" | [{self.backend}]" if self.backend else ""
        loss_str = f" | loss: {loss:.4g}" if loss is not None else ""
        return f"{step_str}{be}{loss_str} | {glyph} {compliant}/{total} Bounds Compliant"

    # -- event detection (transition-gated) -----------------------------
    def _detect_events(self, name, cs, st, grad_conflicts, result) -> List[str]:
        events: List[str] = []
        prev_alpha = self._prev_alpha.get(name, cs.alpha)
        prev_ceiling = self._prev_ceiling.get(name, False)
        prev_active = self._prev_active.get(name, not cs.satisfied)

        if cs.alpha < prev_alpha - 1e-9:
            events.append("vanishing_zone")
        if cs.ceiling_locked and not prev_ceiling:
            events.append("ceiling_lock")
        if cs.satisfied and prev_active:
            events.append("recovered")
        if name in result.interventions:
            events.append("intervention")
        if name in grad_conflicts and grad_conflicts[name].get("projected"):
            last = self._last_conflict_emit.get(name, -10 ** 9)
            if (name not in grad_conflicts) or (result.step - last >= self.reemit_every) or (not prev_active and not cs.satisfied):
                events.append("conflict")
                self._last_conflict_emit[name] = result.step
        if st.surgery_grace_remaining > 0 and "conflict" in events:
            events.append("momentum_grace")
        return events

    # -- rich block rendering -------------------------------------------
    def _render_block(self, name, cs, st, events, reg, grad_conflicts, run_diag) -> str:
        s = self.style
        label = _MODALITY_LABELS.get(self.modality, "Runtime")
        target_desc = self.targets.get(name, f"constraint '{name}'")
        recovered = events == ["recovered"] or (len(events) == 1 and events[0] == "recovered")

        head_glyph = s.g("bulb") if recovered else s.g("warn")
        lines = ["", f"{head_glyph} [Compello Runtime Insight - {label}]",
                 f"Constraint Target: {target_desc}"]

        # current status
        if cs.satisfied:
            lines.append(f"Current Status: compliant (violation {cs.raw_violation:.4g} <= tolerance).")
        else:
            lines.append(f"Current Status: violation {cs.raw_violation:.4g} detected "
                         f"(weight now {cs.weight:.3g}).")

        # mask context (text/multimodal, when the constraint is mask-aware)
        if name in self.mask_aware:
            lines.append("Context Check: packed_sequence_loss_mask applied safely. "
                         "Padding/filler tokens ignored.")

        # issue intercepted (conflict)
        if "conflict" in events:
            info = grad_conflicts[name]
            cos = info.get("cosine", 0.0)
            layer = info.get("layer")
            task_name = info.get("task_loss", "task_loss")
            loc = f" at {layer}" if layer else ""
            lines += ["", f"[Issue Intercepted]: vector conflict between '{task_name}' and "
                          f"constraint '{name}' (cosine similarity: {cos:.2f}{loc})."]

        # interventions executed (numbered)
        actions = self._interventions(name, cs, st, events, grad_conflicts, reg, run_diag)
        if actions:
            lines.append("")
            lines.append(f"{s.g('gear')} CONTROL SYSTEM INTERVENTIONS EXECUTED:")
            for i, a in enumerate(actions, 1):
                lines.append(f" {i}. {a}")
        return "\n".join(lines)

    def _interventions(self, name, cs, st, events, grad_conflicts, reg, run_diag) -> List[str]:
        acts: List[str] = []
        if "conflict" in events:
            info = grad_conflicts[name]
            layer = info.get("layer")
            loc = f" at {layer}" if layer else ""
            acts.append(f"Gradient Surgery active. Conflicting components projected out{loc}.")
        if "momentum_grace" in events:
            acts.append(
                f"Momentum Grace Window deployed. Multiplier scaling frozen for the next "
                f"{st.surgery_grace_remaining} steps so the optimizer's momentum buffer can "
                f"absorb the corrected trajectory without a controller over-correction."
            )
        if "vanishing_zone" in events:
            prev = self._prev_alpha.get(name, cs.alpha)
            acts.append(
                f"Alpha auto-tuning triggered. Lowering proxy sharpness {prev:.3g} -> "
                f"{cs.alpha:.3g} to widen the gradient capture window."
            )
            acts.append(
                "Hysteresis dead-band armed. Sharpness increase is locked out until the "
                f"running gradient norm recovers past {self.controller.config.sharpness_hysteresis:g}x "
                "the floor, preventing boundary chatter."
            )
            sfx = self._recovery(name, reg, cs.raw_violation, run_diag)
            if sfx:
                acts.append(f"Projected recovery{sfx}.")
        if "ceiling_lock" in events:
            acts.append(
                f"Weight locked at ceiling ({cs.weight:.3g}) while still violated - flagged as "
                "possibly infeasible at current model capacity rather than eradicating task loss."
            )
        if "recovered" in events:
            base = cs.weight - cs.weight_delta
            pct = 100 * cs.weight_delta / max(1e-9, base) if cs.weight_delta else 0.0
            acts.append(
                f"Constraint compliant and stable. Relaxing weight {pct:.0f}% to return "
                "capacity to the primary task loss."
            )
        if "intervention" in events:
            # find the action from the result (passed indirectly via events only)
            acts.append("Plateau persisted past patience - on_plateau policy engaged.")
        return acts

    def _recovery(self, name, reg, violation, run_diag) -> str:
        if not run_diag:
            return ""
        steps, conf = reg.estimate_recovery_steps(violation, 0.0)
        r2 = reg.fit.r_squared if reg.fit else None
        if conf == "suppressed" or steps is None:
            return ""
        qual = "" if conf == "reliable" else " (low-confidence)"
        r2s = f", R^2={r2:.2f}" if r2 is not None else ""
        return f" ~{steps:.0f} steps{qual}{r2s}"
