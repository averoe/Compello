"""NumPy reference backend.

This is the load-bearing *testable* backend: it lets the entire portable core
(penalties, controller math, gradient-surgery projection, regression estimator)
be exercised end-to-end without any deep-learning framework installed.

numpy is intentionally an *optional* dependency -- importing this module only
succeeds if numpy is present, exactly like the torch/tf/jax adapters only
import if their framework is present (Section 9 optional-backend pattern).
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - environment without numpy
    _np = None
    _HAS_NUMPY = False


class NumpyBackend:
    name = "numpy"

    @staticmethod
    def is_available() -> bool:
        return _HAS_NUMPY

    # --- construction / inspection --------------------------------------
    def asarray(self, x: Any):
        return _np.asarray(x, dtype=_np.float64) if not _is_np(x) else x

    def to_float(self, x: Any) -> float:
        if _is_np(x):
            return float(_np.asarray(x).mean()) if _np.asarray(x).size > 1 else float(x)
        return float(x)

    def is_native(self, x: Any) -> bool:
        return _is_np(x)

    def shape(self, x: Any) -> tuple:
        return tuple(_np.asarray(x).shape)

    # --- elementwise -----------------------------------------------------
    def maximum(self, a, b):
        return _np.maximum(a, b)

    def minimum(self, a, b):
        return _np.minimum(a, b)

    def abs(self, x):
        return _np.abs(x)

    def exp(self, x):
        return _np.exp(x)

    def log(self, x):
        return _np.log(x)

    def clip(self, x, lo, hi):
        return _np.clip(x, lo, hi)

    def sigmoid(self, x):
        # numerically stable logistic sigmoid
        x = _np.asarray(x, dtype=_np.float64)
        out = _np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + _np.exp(-x[pos]))
        ex = _np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    # --- reductions ------------------------------------------------------
    def sum(self, x, axis=None):
        return _np.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return _np.mean(x, axis=axis)

    def relu(self, x):
        return _np.maximum(x, 0.0)

    # --- linear algebra --------------------------------------------------
    def dot(self, a, b):
        return _np.dot(_np.ravel(a), _np.ravel(b))

    def norm(self, x):
        return _np.linalg.norm(_np.ravel(x))

    # --- shape ops -------------------------------------------------------
    def stack(self, xs, axis=0):
        return _np.stack([_np.asarray(x) for x in xs], axis=axis)

    def flatten(self, x):
        return _np.ravel(x)

    def softmax(self, x, axis=-1):
        x = _np.asarray(x, dtype=_np.float64)
        x = x - _np.max(x, axis=axis, keepdims=True)
        e = _np.exp(x)
        return e / _np.sum(e, axis=axis, keepdims=True)

    def diff(self, x, axis=-1):
        return _np.diff(x, axis=axis)


def _is_np(x: Any) -> bool:
    return _HAS_NUMPY and isinstance(x, _np.ndarray)
