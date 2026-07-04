"""Keras 3 ``keras.ops`` math backend (Section 4.3 / 10.2).

When Keras 3 is installed, this backend implements the shared math surface
strictly in terms of ``keras.ops`` primitives, so a single ``expect()`` penalty
executes identically whichever Keras backend (torch / tensorflow / jax) is
active -- Keras 3 dispatches the ``keras.ops`` call to that backend itself.

It is registered as *available* but not made the default active backend, because
Keras tensors are also native torch/tf/jax tensors and would otherwise contend
with those backends for dispatch. Call ``compello.math.set_backend("keras")`` to
route all penalty math through ``keras.ops`` explicitly.
"""

from __future__ import annotations

from typing import Any

try:
    import keras
    from keras import ops as kops

    _HAS_KERAS = True
except Exception:  # pragma: no cover - environment without keras
    keras = None
    kops = None
    _HAS_KERAS = False


class KerasOpsBackend:
    name = "keras"

    @staticmethod
    def is_available() -> bool:
        return _HAS_KERAS

    def asarray(self, x):
        return kops.convert_to_tensor(x)

    def to_float(self, x):  # pragma: no cover - requires keras
        arr = keras.ops.convert_to_numpy(self.asarray(x))
        return float(arr.mean()) if getattr(arr, "size", 1) > 1 else float(arr)

    def is_native(self, x) -> bool:
        # Deliberately conservative: Keras tensors are backend tensors, so we do
        # not claim ownership for auto-dispatch; this backend is used only when
        # explicitly set active.
        return False

    def shape(self, x):
        return tuple(kops.shape(self.asarray(x)))

    def maximum(self, a, b):
        return kops.maximum(a, b)

    def minimum(self, a, b):
        return kops.minimum(a, b)

    def abs(self, x):
        return kops.absolute(self.asarray(x))

    def exp(self, x):
        return kops.exp(self.asarray(x))

    def log(self, x):
        return kops.log(self.asarray(x))

    def clip(self, x, lo, hi):
        return kops.clip(self.asarray(x), lo, hi)

    def sigmoid(self, x):
        return kops.sigmoid(self.asarray(x))

    def relu(self, x):
        return kops.relu(self.asarray(x))

    def sum(self, x, axis=None):
        return kops.sum(self.asarray(x), axis=axis)

    def mean(self, x, axis=None):
        return kops.mean(self.asarray(x), axis=axis)

    def dot(self, a, b):
        fa = kops.reshape(self.asarray(a), (-1,))
        fb = kops.reshape(self.asarray(b), (-1,))
        return kops.sum(fa * fb)

    def norm(self, x):
        return kops.norm(kops.reshape(self.asarray(x), (-1,)))

    def stack(self, xs, axis=0):
        return kops.stack([self.asarray(x) for x in xs], axis=axis)

    def flatten(self, x):
        return kops.reshape(self.asarray(x), (-1,))

    def softmax(self, x, axis=-1):
        return kops.softmax(self.asarray(x), axis=axis)

    def diff(self, x, axis=-1):
        return kops.diff(self.asarray(x), axis=axis)
