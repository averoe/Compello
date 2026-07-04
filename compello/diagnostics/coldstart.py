"""Noise-aware cold-start relaxation (Section 5.3).

Early in training the model is dominated by initialization noise; enforcing
strict constraints immediately risks corrupting (pre)trained weights before the
model has stabilised. Compello monitors the variance of raw output/constraint
gradients across a sliding early-training window and defers full enforcement
until that variance falls, applying a soft-start relaxation factor in the
meantime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class ColdStartState:
    relaxation: float   # multiplier applied to constraint weights (1.0 = full)
    stabilised: bool
    variance: float


class ColdStartMonitor:
    def __init__(
        self,
        window: int = 200,
        soft_start_steps: int = 2000,
        variance_threshold: float = 1.0,
        min_relaxation: float = 0.25,
    ):
        self.window = window
        self.soft_start_steps = soft_start_steps
        self.variance_threshold = variance_threshold
        self.min_relaxation = min_relaxation
        self._grad_norms: Deque[float] = deque(maxlen=window)
        self._step = 0
        self._stabilised = False

    def update(self, grad_norm: float) -> ColdStartState:
        self._grad_norms.append(float(grad_norm))
        self._step += 1
        var = _variance(self._grad_norms)

        # stabilised once variance drops below threshold OR the soft-start
        # window elapses -- whichever first, then it stays stabilised.
        if not self._stabilised and (
            (len(self._grad_norms) >= 2 and var < self.variance_threshold)
            or self._step >= self.soft_start_steps
        ):
            self._stabilised = True

        if self._stabilised:
            return ColdStartState(relaxation=1.0, stabilised=True, variance=var)

        # linear ramp of enforcement from min_relaxation -> 1.0 across the window
        frac = min(1.0, self._step / max(1, self.soft_start_steps))
        relax = self.min_relaxation + (1.0 - self.min_relaxation) * frac
        return ColdStartState(relaxation=relax, stabilised=False, variance=var)


def _variance(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / (n - 1)
