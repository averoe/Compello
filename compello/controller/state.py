"""Per-constraint controller state (Sections 4.2, 4.3, 4.4, 6.2).

Everything the controller mutates lives here as plain Python scalars/lists so it
can be (a) executed identically on any backend -- the math is scalar recurrence
relations, not framework tensor ops (Section 3.2 backend note) -- and (b)
serialised into any framework's native checkpoint format (Section 6.2).

``to_dict``/``from_dict`` are the checkpoint surface. Cooper checkpoints only
multipliers + dual-optimizer state; the fields below (EMA buffers, rolling
baseline accumulator, stability counters, adaptive sharpness + hysteresis arm)
are exactly the state that only exists because of Compello's EMA/PID layer and
would otherwise restart cold on resume, momentarily reintroducing the
derivative-kick risk (Section 4.2) the layer exists to prevent.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class ConstraintState:
    def __init__(self, name: str, initial_weight: float = 1.0, baseline_window: int = 200):
        self.name = name
        self.weight = float(initial_weight)

        # EMA buffers (4.2)
        self.smoothed: Optional[float] = None   # slow EMA -> PID input
        self.fast: Optional[float] = None        # fast EMA -> spike detector

        # rolling baseline accumulator for the fast-EMA detector (4.2)
        self.baseline_window = int(baseline_window)
        self.fast_history: Deque[float] = deque(maxlen=self.baseline_window)

        # spike / override tracking (4.2)
        self.consecutive_elevated = 0
        self.override_remaining = 0

        # PID accumulators (3.2 / 4.2)
        self.integral = 0.0
        self.prev_error = 0.0

        # adaptive sharpness + hysteresis arm (4.3)
        self.alpha = 1.0
        self.sharpness_armed = True  # armed to scale down when grad norm dips

        # stability window (3.2 relaxation)
        self.stable_steps = 0

        # weight-ceiling / feasibility flags (4.4)
        self.ceiling_locked = False
        # on_plateau intervention tracking (3.2)
        self.plateau_steps = 0
        self.attempts_used = 0

        # momentum-aware grace window (4.7)
        self.surgery_grace_remaining = 0

        # gradient-accumulation freeze (1.4): micro-batch violation accumulator
        self.micro_accum = 0.0
        self.micro_count = 0

        # hysteresis dead-band patience (1.5): steps spent locked between the
        # scale-down trigger and the scale-up re-arm band
        self.deadband_steps = 0

        self.last_raw_violation = 0.0
        self.last_weight_delta = 0.0
        self.steps_since_intervention = 0

    # -- baseline statistics (4.2) --------------------------------------
    def baseline_stats(self, exclude_recent: int) -> Optional[tuple]:
        """Return ``(mean, std)`` of the fast-EMA history, excluding the most
        recent ``exclude_recent`` entries (the candidate spike window) so the
        baseline is not dragged up by the spike being evaluated against it.
        """
        hist: List[float] = list(self.fast_history)
        if exclude_recent > 0:
            hist = hist[:-exclude_recent] if exclude_recent < len(hist) else []
        if len(hist) < 2:
            return None
        n = len(hist)
        mean = sum(hist) / n
        var = sum((h - mean) ** 2 for h in hist) / (n - 1)
        return mean, var ** 0.5

    # -- checkpointing (6.2) --------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "smoothed": self.smoothed,
            "fast": self.fast,
            "baseline_window": self.baseline_window,
            "fast_history": list(self.fast_history),
            "consecutive_elevated": self.consecutive_elevated,
            "override_remaining": self.override_remaining,
            "integral": self.integral,
            "prev_error": self.prev_error,
            "alpha": self.alpha,
            "sharpness_armed": self.sharpness_armed,
            "stable_steps": self.stable_steps,
            "ceiling_locked": self.ceiling_locked,
            "plateau_steps": self.plateau_steps,
            "attempts_used": self.attempts_used,
            "surgery_grace_remaining": self.surgery_grace_remaining,
            "micro_accum": self.micro_accum,
            "micro_count": self.micro_count,
            "deadband_steps": self.deadband_steps,
            "last_raw_violation": self.last_raw_violation,
            "last_weight_delta": self.last_weight_delta,
            "steps_since_intervention": self.steps_since_intervention,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintState":
        s = cls(d["name"], initial_weight=d["weight"], baseline_window=d["baseline_window"])
        s.smoothed = d["smoothed"]
        s.fast = d["fast"]
        s.fast_history = deque(d["fast_history"], maxlen=s.baseline_window)
        s.consecutive_elevated = d["consecutive_elevated"]
        s.override_remaining = d["override_remaining"]
        s.integral = d["integral"]
        s.prev_error = d["prev_error"]
        s.alpha = d["alpha"]
        s.sharpness_armed = d["sharpness_armed"]
        s.stable_steps = d["stable_steps"]
        s.ceiling_locked = d["ceiling_locked"]
        s.plateau_steps = d.get("plateau_steps", 0)
        s.attempts_used = d.get("attempts_used", 0)
        s.surgery_grace_remaining = d["surgery_grace_remaining"]
        s.micro_accum = d.get("micro_accum", 0.0)
        s.micro_count = d.get("micro_count", 0)
        s.deadband_steps = d.get("deadband_steps", 0)
        s.last_raw_violation = d["last_raw_violation"]
        s.last_weight_delta = d["last_weight_delta"]
        s.steps_since_intervention = d["steps_since_intervention"]
        return s
