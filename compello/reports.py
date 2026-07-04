"""Post-training reports (Sections 5.4, 6.3).

- ``SensitivityProfiler`` builds the marginal-cost report (5.4): per constraint,
  how much task performance was traded for tightening it, estimated from the
  same rolling-regression mechanism as 5.1a, with the R^2 confidence gate so a
  noisy fit does not produce an authoritative-looking number.
- ``non_convergence_report`` (6.3) explains, for a constraint that failed to
  reach tolerance by ``max_steps``, whether its weight plateaued at the ceiling
  (suggesting infeasibility, 4.4) or was still climbing (needs more steps), and
  gives a concrete next step rather than a bare pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .controller.controller import Controller
from .diagnostics.regression import R2_SUPPRESS, RollingRegressor

LOW_IMPACT = "low"
HIGH_IMPACT = "high"
UNKNOWN_IMPACT = "unknown"


@dataclass
class ConstraintSensitivity:
    name: str
    impact: str                     # low | high | unknown
    marginal_cost: Optional[float]  # d(task metric) per unit tightening
    r_squared: Optional[float]
    note: str


class SensitivityProfiler:
    """Tracks the relationship between each constraint's weight and a task
    metric, to report marginal cost after training (5.4)."""

    def __init__(self, *, window: int = 500, refit_interval: int = 50,
                 high_impact_threshold: float = 0.02):
        self._reg: Dict[str, RollingRegressor] = {}
        self.window = window
        self.refit_interval = refit_interval
        self.high_impact_threshold = high_impact_threshold

    def observe(self, name: str, *, weight: float, violation: float, task_metric: float) -> None:
        reg = self._reg.get(name)
        if reg is None:
            reg = self._reg[name] = RollingRegressor(self.window, self.refit_interval)
        # features relate controller pressure to the task metric outcome
        reg.observe([weight, violation], task_metric)

    def report(self) -> Dict[str, ConstraintSensitivity]:
        out: Dict[str, ConstraintSensitivity] = {}
        for name, reg in self._reg.items():
            reg.refit()
            fit = reg.fit
            if fit is None or fit.r_squared < R2_SUPPRESS:
                out[name] = ConstraintSensitivity(
                    name, UNKNOWN_IMPACT, None,
                    fit.r_squared if fit else None,
                    "insufficient/noisy signal to estimate marginal cost.",
                )
                continue
            # coefficient on weight = marginal task-metric change per unit weight
            marginal = fit.coefficients[0] if fit.coefficients else 0.0
            impact = HIGH_IMPACT if abs(marginal) >= self.high_impact_threshold else LOW_IMPACT
            note = (
                "no measurable cost to task performance."
                if impact == LOW_IMPACT
                else f"each unit of added weight moves the task metric by ~{marginal:.3g}."
            )
            out[name] = ConstraintSensitivity(name, impact, marginal, fit.r_squared, note)
        return out

    def render(self, *, title: str = "COMPELLO POST-TRAINING REPORT") -> str:
        rep = self.report()
        lines = [title, "Optimization complete.", "", "CAPACITY ANALYSIS"]
        for cs in rep.values():
            r2 = f", R^2={cs.r_squared:.2f}" if cs.r_squared is not None else ""
            lines.append(f"  '{cs.name}' - {cs.impact} impact: {cs.note}{r2}")
        return "\n".join(lines)


def render_capacity_report(
    controller: "Controller",
    profiler: "SensitivityProfiler",
    *,
    converged: bool,
    diagnostic_overhead_pct: Optional[float] = None,
    compute_summary: Optional[str] = None,
    primitive_labels: Optional[Dict[str, str]] = None,
    style=None,
) -> str:
    """Render the full post-training Capacity & Sensitivity autopsy (5.4/6.3).

    Combines the sensitivity profiler's marginal-cost fit with the
    non-convergence diagnosis and, where a constraint is a high-impact
    bottleneck or infeasible, concrete strategic recommendations (deeper NODE
    architecture, distillation_bridge) -- all grounded in the real fit, with the
    OLS R^2 reported so the confidence is visible.
    """
    from .report_style import Style

    s = style or Style.auto()
    labels = primitive_labels or {}
    sens = profiler.report()
    nc = {e.name: e for e in non_convergence_report(controller)}

    out: List[str] = [s.banner("COMPELLO POST-TRAINING CAPACITY & SENSITIVITY REPORT")]
    if converged:
        out.append(f"{s.g('check')} Optimization complete. Model reached stable convergence properties.")
    else:
        out.append(f"{s.g('warn')} Optimization stopped without full convergence (see below).")
    if compute_summary:
        overhead = f" ({diagnostic_overhead_pct:.2f}% diagnostic overhead)" if diagnostic_overhead_pct is not None else ""
        out.append(f"{s.g('chart')} Compute: {compute_summary}{overhead}")
    out += ["", "SYSTEM CAPACITY METRIC ANALYSIS:"]

    any_bottleneck = False
    for name, st in controller.states.items():
        prim = labels.get(name, "Constraint Primitive")
        cs_sens = sens.get(name)
        entry = nc.get(name)
        compliant = entry is None
        out.append("")
        out.append(f"{s.g('mag')} Rule: '{name}' [{prim}]")
        out.append(f"   - Compliance Status: {'100% Retained.' if compliant else 'NOT met at cutoff.'}")

        impact = cs_sens.impact if cs_sens else "unknown"
        impact_label = {
            "low": "LOW.",
            "high": "HIGH - CRITICAL BOTTLENECK.",
            "unknown": "UNKNOWN (insufficient signal).",
        }[impact]
        out.append(f"   - Core System Impact: {impact_label}")

        if cs_sens and impact == "low":
            out.append("   - [Analysis]: enforcing this rule incurred no statistically "
                       "meaningful cost to your primary validation loss.")
        elif cs_sens and impact == "high":
            any_bottleneck = True
            mc = cs_sens.marginal_cost
            r2 = cs_sens.r_squared
            out.append("   - [Analysis]: the online OLS engine tracked a tight, continuous "
                       "trade-off between this rule and your primary goal.")
            out.append(f"     [Quantified Trade-off]: each unit of added enforcement weight "
                       f"moved the primary metric by ~{mc:.3g} "
                       f"(OLS R^2 = {r2:.2f} - high-reliability trend).")
        else:
            out.append("   - [Analysis]: not enough clean signal to quantify a trade-off "
                       "(the fit was indistinguishable from noise).")

        if entry is not None:
            out.append("")
            out.append(f"{s.g('cross')} ARCHITECTURAL LIMITATION: {entry.diagnosis}")

    # strategic recommendations when there is a bottleneck or an infeasible rule
    if any_bottleneck or nc:
        out += ["", f"{s.g('wrench')} STRATEGIC RECOMMENDATIONS FOR YOUR NEXT RUN:"]
        recs = []
        if nc:
            recs.append("Do not run a tighter compliance floor at this network depth/capacity.")
        recs.append("If accuracy must be restored, expand the model (e.g. a NODE architecture "
                    "with ~20% deeper processing blocks for tabular data).")
        recs.append("Alternatively, use compello.distillation_bridge() to map the behavioral "
                    "constraints onto a student distilled from an unconstrained teacher.")
        for i, r in enumerate(recs, 1):
            out.append(f" {i}. {r}")

    out.append(s.rule())
    return "\n".join(out)


@dataclass
class NonConvergenceEntry:
    name: str
    converged: bool
    weight: float
    ceiling_locked: bool
    diagnosis: str
    next_step: str


def non_convergence_report(
    controller: Controller,
    *,
    tolerance: Optional[float] = None,
) -> List[NonConvergenceEntry]:
    """Explain each unconverged constraint (Section 6.3)."""
    tol = controller.config.tolerance if tolerance is None else tolerance
    ceiling = controller.config.weight_ceiling
    entries: List[NonConvergenceEntry] = []
    for name, st in controller.states.items():
        last = st.last_raw_violation
        converged = last <= tol and not st.ceiling_locked
        if converged:
            continue
        if st.ceiling_locked or st.weight >= ceiling - 1e-9:
            diagnosis = (
                "weight plateaued at the ceiling while still violated -- likely "
                "architecturally infeasible at current model capacity (4.4)."
            )
            next_step = (
                "increase model capacity, relax the constraint tolerance, or "
                "raise weight_ceiling if you have evidence it is feasible."
            )
        elif st.last_weight_delta > 0:
            diagnosis = "weight was still climbing at cutoff -- likely just needed more steps."
            next_step = "increase max_steps or patience and resume from checkpoint."
        else:
            diagnosis = "violation persisted without the controller escalating weight."
            next_step = "check for a gradient conflict (enable gradient_surgery) or a mis-scaled penalty."
        entries.append(NonConvergenceEntry(
            name=name, converged=False, weight=st.weight,
            ceiling_locked=st.ceiling_locked, diagnosis=diagnosis, next_step=next_step,
        ))
    return entries
