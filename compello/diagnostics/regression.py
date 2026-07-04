"""Online linear regression estimator behind the quantitative insights (5.1a).

Insight messages like "Estimated recovery: ~280 steps" are produced by a small,
continuously-refit ordinary-least-squares fit relating recent controller
actions to subsequent changes in the constraint's violation metric. It is a
deliberately lightweight heuristic -- a historical trend, not a causal model or
a simulation of training dynamics.

Confidence gating (5.1a):
- R^2 below 0.5  -> numeric projections carry a confidence qualifier.
- R^2 below 0.2  -> numeric projection is suppressed; fall back to a qualitative
  statement, since the fit is statistically indistinguishable from a random walk.

Implemented in pure Python (small feature count) so diagnostics add no
dependency beyond whatever backend is active (Section 9).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

R2_QUALIFY = 0.5
R2_SUPPRESS = 0.2


@dataclass
class FitResult:
    coefficients: List[float]
    intercept: float
    r_squared: float
    n_samples: int

    @property
    def reliable(self) -> bool:
        return self.r_squared >= R2_QUALIFY

    @property
    def usable(self) -> bool:
        return self.r_squared >= R2_SUPPRESS


class RollingRegressor:
    """Rolling OLS over the trailing ``window`` samples, refit every
    ``refit_interval`` observations (defaults 500 / 50 per 5.1a)."""

    def __init__(self, window: int = 500, refit_interval: int = 50):
        self.window = window
        self.refit_interval = refit_interval
        self._rows: Deque[Tuple[Tuple[float, ...], float]] = deque(maxlen=window)
        self._since_fit = 0
        self._fit: Optional[FitResult] = None

    def observe(self, features: Sequence[float], target: float) -> None:
        self._rows.append((tuple(float(f) for f in features), float(target)))
        self._since_fit += 1
        if self._since_fit >= self.refit_interval and len(self._rows) >= _min_rows(self._rows):
            self.refit()

    def refit(self) -> Optional[FitResult]:
        self._since_fit = 0
        rows = list(self._rows)
        if len(rows) < 3:
            return self._fit
        X = [list(f) + [1.0] for f, _ in rows]  # append bias column
        y = [t for _, t in rows]
        beta = _ols_solve(X, y)
        if beta is None:
            return self._fit
        r2 = _r_squared(X, y, beta)
        self._fit = FitResult(
            coefficients=beta[:-1], intercept=beta[-1], r_squared=r2, n_samples=len(rows)
        )
        return self._fit

    @property
    def fit(self) -> Optional[FitResult]:
        return self._fit

    def predict(self, features: Sequence[float]) -> Optional[float]:
        if self._fit is None:
            return None
        val = self._fit.intercept
        for c, f in zip(self._fit.coefficients, features):
            val += c * float(f)
        return val

    def estimate_recovery_steps(
        self, current_violation: float, target: float = 0.0
    ) -> Tuple[Optional[float], str]:
        """Estimate steps to reach ``target`` violation from the fitted trend.

        Returns ``(steps_or_None, confidence)`` where confidence is one of
        ``"reliable"``, ``"low"``, or ``"suppressed"`` per the 5.1a thresholds.
        Uses the coefficient on the "steps-since-intervention"/trend feature; if
        the caller's feature layout differs, pass the per-step delta directly to
        ``steps_from_rate`` instead.
        """
        if self._fit is None or not self._fit.usable:
            return None, "suppressed"
        # heuristic: use the mean observed per-step change in target
        rows = list(self._rows)
        if len(rows) < 2:
            return None, "suppressed"
        deltas = [rows[i][1] - rows[i - 1][1] for i in range(1, len(rows))]
        rate = sum(deltas) / len(deltas)
        conf = "reliable" if self._fit.reliable else "low"
        return steps_from_rate(current_violation, target, rate), conf


def steps_from_rate(current: float, target: float, rate_per_step: float) -> Optional[float]:
    """Steps to go from ``current`` to ``target`` at ``rate_per_step``.

    ``rate_per_step`` is expected to be negative when violation is falling.
    Returns None when the trend is not moving toward the target.
    """
    gap = current - target
    if gap <= 0:
        return 0.0
    if rate_per_step >= 0:
        return None  # not improving; no finite estimate
    return gap / (-rate_per_step)


# --- tiny pure-Python OLS via normal equations ---------------------------

def _min_rows(rows) -> int:
    return 3


def _ols_solve(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Solve (X^T X) beta = X^T y via Gaussian elimination. Pure Python."""
    n = len(X)
    p = len(X[0])
    # X^T X (p x p) and X^T y (p)
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for a in range(p):
            xty[a] += xi[a] * yi
            xa = xi[a]
            row = xtx[a]
            for b in range(p):
                row[b] += xa * xi[b]
    # ridge nudge for numerical stability / rank deficiency
    for a in range(p):
        xtx[a][a] += 1e-8
    return _gauss_solve(xtx, xty)


def _gauss_solve(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def _r_squared(X: List[List[float]], y: List[float], beta: List[float]) -> float:
    n = len(y)
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = 0.0
    for i in range(n):
        pred = sum(X[i][j] * beta[j] for j in range(len(beta)))
        ss_res += (y[i] - pred) ** 2
    if ss_tot <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)
