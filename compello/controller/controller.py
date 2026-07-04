"""The adaptive constraint-weight controller (Section 3.2).

Watches live violation every step and adjusts each constraint's weight. Four
swappable strategies: ``fixed``, ``linear_ramp``, ``adaptive_pid`` (the default,
with the full EMA/PID/ceiling machinery), and ``dual_ascent`` (the classic
Lagrangian baseline, deliberately without the smoothing layer so it can be
compared fairly against ``adaptive_pid``).

The controller math is framework-neutral scalar recurrence; only *how* the
state is stored differs per backend (Section 10). Here it is plain Python.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ceiling import apply_ceiling
from .ema import update_ema_layer
from .pid import PIDGains, pid_step
from .sharpness import update_sharpness
from .state import ConstraintState

FIXED = "fixed"
LINEAR_RAMP = "linear_ramp"
ADAPTIVE_PID = "adaptive_pid"
DUAL_ASCENT = "dual_ascent"


@dataclass
class ControllerConfig:
    strategy: str = ADAPTIVE_PID
    tolerance: float = 1e-3
    patience: int = 500
    max_steps: int = 50000

    # EMA layer (4.2)
    ema_decay: float = 0.97
    ema_fast_decay: float = 0.7
    ema_override_steps: int = 5
    ema_baseline_window: int = 200

    # weight ceiling (4.4)
    weight_ceiling: float = 25.0

    # sharpness hysteresis (4.3) + dead-band patience decay (1.5)
    sharpness_hysteresis: float = 1.5
    sharpness_g_floor: float = 1e-4
    sharpness_patience: int = 0   # steps stuck in the dead-band before re-arm decay (0 = off)

    # PID gains
    gains: PIDGains = field(default_factory=PIDGains)
    weight_lr: float = 0.1        # scales PID output into a weight delta
    relax_factor: float = 0.85    # multiplicative relaxation when stable
    dual_lr: float = 0.05         # dual-ascent step size

    # linear_ramp
    ramp_target: float = 10.0
    ramp_steps: int = 10000

    on_plateau: str = "report_infeasible"  # "reduce_lr" | "rollback" | "report_infeasible"
    max_attempts: int = 3       # intervention budget before reporting infeasible (3.2)

    # gradient-accumulation freeze (1.4): number of micro-batches per macro step.
    # With >1, violations are averaged across micro-batches and the multiplier is
    # only updated at the macro boundary.
    accumulation_steps: int = 1

    # log-space multiplier stability (1.3): keep the effective multiplier update
    # numerically stable when violations are tiny (FP16/AMP underflow regime).
    log_space_stability: bool = False

    # opt-in, version-fragile direct optimizer momentum-buffer surgery (4.7).
    # Off by default; the adapter reads this flag -- the controller only plumbs it.
    aggressive_momentum_correction: bool = False


@dataclass
class ConstraintStep:
    name: str
    weight: float
    raw_violation: float
    smoothed_violation: float
    satisfied: bool
    stable: bool
    weight_delta: float
    ceiling_locked: bool
    alpha: float


@dataclass
class StepResult:
    step: int
    per_constraint: Dict[str, ConstraintStep]
    all_satisfied: bool
    converged: bool          # all satisfied for >= patience
    plateau_flags: List[str] # constraints locked at ceiling & still violated
    interventions: Dict[str, str] = field(default_factory=dict)  # name -> action (3.2)
    should_stop: bool = False  # convergence reached or max_steps hit


class Controller:
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.states: Dict[str, ConstraintState] = {}
        self.step_index = 0
        self._converged_run = 0  # consecutive steps with all constraints satisfied

    # -- registration ----------------------------------------------------
    def register(self, name: str, initial_weight: float = 1.0) -> ConstraintState:
        st = ConstraintState(
            name, initial_weight=initial_weight,
            baseline_window=self.config.ema_baseline_window,
        )
        st.alpha = 1.0
        self.states[name] = st
        return st

    def register_assertions(self, assertions) -> None:
        for a in assertions:
            self.register(a.name, initial_weight=a.initial_weight)

    # -- gradient-accumulation freeze (1.4) -----------------------------
    def micro_step(self, violations: Dict[str, float]) -> Optional[StepResult]:
        """Accumulate a micro-batch's violations without touching multipliers.

        Under ``gradient_accumulation_steps > 1``, calling ``micro_step`` once
        per micro-batch buffers the raw violations; the controller state and
        multipliers are frozen until ``accumulation_steps`` micro-batches have
        been seen, at which point the averaged violation drives exactly one real
        ``step`` at the macro-batch boundary. Returns the ``StepResult`` at the
        boundary, else ``None``.

        This prevents multiplier poisoning from per-micro-batch updates: the
        controller sees the same signal it would on a single large batch.
        """
        for name, v in violations.items():
            st = self.states.get(name) or self.register(name)
            fv = float(v)
            if not _math.isfinite(fv):
                fv = st.last_raw_violation
            st.micro_accum += fv
            st.micro_count += 1

        boundary = all(
            self.states[n].micro_count >= self.config.accumulation_steps
            for n in violations
        )
        if not boundary:
            return None

        averaged = {}
        for name in violations:
            st = self.states[name]
            averaged[name] = st.micro_accum / max(1, st.micro_count)
            st.micro_accum = 0.0
            st.micro_count = 0
        return self.step(averaged)

    # -- main update -----------------------------------------------------
    def step(
        self,
        violations: Dict[str, float],
        *,
        proxy_grad_norms: Optional[Dict[str, float]] = None,
        metric_satisfied: Optional[Dict[str, bool]] = None,
    ) -> StepResult:
        cfg = self.config
        proxy_grad_norms = proxy_grad_norms or {}
        metric_satisfied = metric_satisfied or {}
        per: Dict[str, ConstraintStep] = {}
        plateau_flags: List[str] = []
        interventions: Dict[str, str] = {}

        for name, raw in violations.items():
            st = self.states.get(name) or self.register(name)
            raw = float(raw)
            # NaN/inf guard: a non-finite violation (e.g. from a diverged step)
            # must not poison the EMA/PID state. Carry forward the last good
            # value so the controller degrades gracefully instead of exploding.
            if not _math.isfinite(raw):
                raw = st.last_raw_violation

            # sharpness tuning for proxy constraints, if grad norm supplied
            if name in proxy_grad_norms:
                update_sharpness(
                    st, proxy_grad_norms[name],
                    metric_satisfied.get(name, raw <= cfg.tolerance),
                    g_floor=cfg.sharpness_g_floor,
                    sharpness_hysteresis=cfg.sharpness_hysteresis,
                    sharpness_patience=cfg.sharpness_patience,
                )

            prev_weight = st.weight
            satisfied = raw <= cfg.tolerance

            if satisfied:
                st.stable_steps += 1
            else:
                st.stable_steps = 0
            stable = st.stable_steps >= cfg.patience

            smoothed = raw
            if cfg.strategy == FIXED:
                new_weight = st.weight  # unchanged from initial

            elif cfg.strategy == LINEAR_RAMP:
                frac = min(1.0, self.step_index / max(1, cfg.ramp_steps))
                new_weight = st.weight + frac * (cfg.ramp_target - st.weight) * (
                    1.0 / max(1, cfg.ramp_steps)
                )
                # simpler faithful ramp: interpolate from initial toward target
                new_weight = min(cfg.ramp_target, prev_weight + cfg.ramp_target / cfg.ramp_steps)

            elif cfg.strategy == DUAL_ASCENT:
                # classic Lagrangian multiplier ascent on the RAW signal (no EMA)
                new_weight = max(0.0, st.weight + cfg.dual_lr * raw)
                new_weight = apply_ceiling(
                    st, new_weight, weight_ceiling=cfg.weight_ceiling,
                    still_violated=not satisfied,
                )

            else:  # ADAPTIVE_PID
                smoothed = update_ema_layer(
                    st, raw,
                    ema_decay=cfg.ema_decay,
                    ema_fast_decay=cfg.ema_fast_decay,
                    ema_override_steps=cfg.ema_override_steps,
                )
                if stable and st.surgery_grace_remaining == 0:
                    # satisfied & stable: relax weight, bleed the integral so it
                    # stops fighting the task loss unnecessarily.
                    new_weight = prev_weight * cfg.relax_factor
                    st.integral *= cfg.relax_factor
                else:
                    u = pid_step(st, smoothed, cfg.gains)
                    if st.surgery_grace_remaining > 0:
                        # momentum-aware grace window (4.7): suppress escalation
                        # while optimizer momentum still carries the old direction
                        st.surgery_grace_remaining -= 1
                        u = min(u, 0.0)
                    if cfg.log_space_stability:
                        # AMP/FP16 underflow guard (1.3): update the multiplier in
                        # log-space so it stays strictly positive and tiny updates
                        # never underflow to zero. d(log w) = dw / w reproduces the
                        # additive update to first order while being underflow-safe.
                        log_w = _math.log(max(prev_weight, 1e-12))
                        log_w += cfg.weight_lr * u / max(prev_weight, 1e-6)
                        new_weight = _math.exp(min(max(log_w, -30.0), 30.0))
                    else:
                        new_weight = prev_weight + cfg.weight_lr * u
                new_weight = apply_ceiling(
                    st, new_weight, weight_ceiling=cfg.weight_ceiling,
                    still_violated=not satisfied,
                )

            st.weight = float(new_weight)
            st.last_weight_delta = st.weight - prev_weight
            st.steps_since_intervention = (
                0 if abs(st.last_weight_delta) > 1e-9 else st.steps_since_intervention + 1
            )

            if st.ceiling_locked and not satisfied:
                plateau_flags.append(name)
                st.plateau_steps += 1
            else:
                st.plateau_steps = 0

            # on_plateau intervention: once a constraint has been locked at the
            # ceiling and still violated for `patience` steps, emit the
            # configured action; after `max_attempts` interventions escalate to
            # report_infeasible (3.2).
            if st.plateau_steps >= cfg.patience:
                st.plateau_steps = 0
                if st.attempts_used < cfg.max_attempts:
                    st.attempts_used += 1
                    interventions[name] = cfg.on_plateau
                else:
                    interventions[name] = "report_infeasible"

            per[name] = ConstraintStep(
                name=name, weight=st.weight, raw_violation=raw,
                smoothed_violation=smoothed, satisfied=satisfied, stable=stable,
                weight_delta=st.last_weight_delta, ceiling_locked=st.ceiling_locked,
                alpha=st.alpha,
            )

        all_satisfied = all(c.satisfied for c in per.values()) and len(per) > 0
        if all_satisfied:
            self._converged_run += 1
        else:
            self._converged_run = 0
        converged = self._converged_run >= cfg.patience

        self.step_index += 1
        should_stop = converged or self.step_index >= cfg.max_steps
        return StepResult(
            step=self.step_index - 1, per_constraint=per,
            all_satisfied=all_satisfied, converged=converged,
            plateau_flags=plateau_flags, interventions=interventions,
            should_stop=should_stop,
        )

    def engage_surgery(self, name: str, beta1: float) -> None:
        """Open a momentum-aware grace window for a constraint (Section 4.7).

        Duration ~ one optimizer first-moment time constant ``1/(1-beta1)``.
        """
        st = self.states.get(name)
        if st is None:
            return
        st.surgery_grace_remaining = max(1, int(round(1.0 / max(1e-6, (1.0 - beta1)))))

    @property
    def weights(self) -> Dict[str, float]:
        return {n: s.weight for n, s in self.states.items()}

    # -- checkpointing (6.2) --------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "converged_run": self._converged_run,
            "states": {n: s.to_dict() for n, s in self.states.items()},
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.step_index = d["step_index"]
        self._converged_run = d["converged_run"]
        self.states = {n: ConstraintState.from_dict(sd) for n, sd in d["states"].items()}
