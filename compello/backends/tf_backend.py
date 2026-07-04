"""TensorFlow / Keras backend adapter (Sections 4.1b, 4.6b, 4.7b, 10.3).

Requires ``tensorflow``. Provides:
  * ``TFBackend`` -- the ``compello.math`` implementation for ``tf.Tensor``.
  * ``ConstraintTape`` -- wraps a ``tf.GradientTape`` and exposes
    ``steer_gradients`` after ``tape.gradient(...)`` but before
    ``apply_gradients(...)``, mirroring the PyTorch post-backward/pre-step
    timing (10.3).
  * ``sync_violations`` via ``tf.distribute.Strategy.reduce`` (4.1b).
  * ``CompelloKerasCallback`` -- a ``keras.callbacks.Callback`` (3.3).

Controller state that lives across steps is stored in ``tf.Variable(...,
trainable=False)`` and updated with ``.assign_add`` so AutoGraph traces it as a
real graph node rather than baking a Python constant (4.6b).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import BackendNotAvailableError

try:
    import tensorflow as tf

    _HAS_TF = True
except Exception:  # pragma: no cover
    tf = None
    _HAS_TF = False


def _require():
    if not _HAS_TF:
        raise BackendNotAvailableError(
            "compello.backends.tf requires TensorFlow. Install with "
            "`pip install compello[tensorflow]`."
        )


class TFBackend:
    name = "tensorflow"

    @staticmethod
    def is_available() -> bool:
        return _HAS_TF

    def asarray(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32) if not tf.is_tensor(x) else tf.cast(x, tf.float32)

    def to_float(self, x):
        t = self.asarray(x)
        return float(tf.reduce_mean(t)) if tf.size(t) > 1 else float(tf.reshape(t, [-1])[0])

    def is_native(self, x) -> bool:
        return _HAS_TF and tf.is_tensor(x)

    def shape(self, x):
        return tuple(self.asarray(x).shape)

    def maximum(self, a, b):
        return tf.maximum(self.asarray(a), self.asarray(b))

    def minimum(self, a, b):
        return tf.minimum(self.asarray(a), self.asarray(b))

    def abs(self, x):
        return tf.abs(self.asarray(x))

    def exp(self, x):
        return tf.exp(self.asarray(x))

    def log(self, x):
        return tf.math.log(self.asarray(x))

    def clip(self, x, lo, hi):
        return tf.clip_by_value(self.asarray(x), lo, hi)

    def sigmoid(self, x):
        return tf.sigmoid(self.asarray(x))

    def relu(self, x):
        return tf.nn.relu(self.asarray(x))

    def sum(self, x, axis=None):
        return tf.reduce_sum(self.asarray(x), axis=axis)

    def mean(self, x, axis=None):
        return tf.reduce_mean(self.asarray(x), axis=axis)

    def dot(self, a, b):
        return tf.reduce_sum(tf.reshape(self.asarray(a), [-1]) * tf.reshape(self.asarray(b), [-1]))

    def norm(self, x):
        return tf.norm(tf.reshape(self.asarray(x), [-1]))

    def stack(self, xs, axis=0):
        return tf.stack([self.asarray(x) for x in xs], axis=axis)

    def flatten(self, x):
        return tf.reshape(self.asarray(x), [-1])

    def softmax(self, x, axis=-1):
        return tf.nn.softmax(self.asarray(x), axis=axis)

    def diff(self, x, axis=-1):
        t = self.asarray(x)
        n = t.shape[axis]
        front = tf.gather(t, range(1, n), axis=axis)
        back = tf.gather(t, range(0, n - 1), axis=axis)
        return front - back


class TFAdapter:
    name = "tensorflow"

    def __init__(self, model=None, optimizer=None, strategy=None):
        _require()
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy

    # -- 4.1b: strategy.reduce, one collective per step ----------------
    def sync_violations(self, local_violations: Dict[str, float]) -> Dict[str, float]:
        strategy = self.strategy or tf.distribute.get_strategy()
        # default (no-op) strategy returns identity; real strategies reduce.
        if isinstance(strategy, tf.distribute.get_strategy().__class__) and \
                strategy.num_replicas_in_sync == 1:
            return dict(local_violations)
        from .sync import batched_sync

        def reduce_fn(values):
            stacked = tf.stack(values)
            synced = strategy.reduce(tf.distribute.ReduceOp.MEAN, stacked, axis=None)
            return [float(synced[i]) for i in range(len(values))]

        return batched_sync(local_violations, reduce_fn)

    def optimizer_beta1(self) -> Optional[float]:
        if self.optimizer is None:
            return None
        cfg = self.optimizer.get_config()
        if "beta_1" in cfg:
            return float(cfg["beta_1"])
        if "momentum" in cfg:
            return float(cfg["momentum"])
        return None

    # -- 4.7b: aggressive direct momentum-buffer surgery (opt-in) ------
    def project_momentum_buffers(self, task_grads_by_var: Dict[str, Any]) -> int:
        """Project the conflicting component out of a Keras optimizer's
        first-moment slot variables (Section 4.7b, ``aggressive_momentum_correction``).

        Keras 3 exposes first-moment state through ``optimizer.variables``, keyed
        by variable path rather than a fixed dict key -- the same fragility as
        PyTorch's ``exp_avg``. This matches momentum variables (path contains
        ``momentum``/``/m``) to the provided per-variable task gradients by name
        substring and removes the task-direction component in place where they
        conflict. Best-effort and version-fragile by nature; opt-in only.
        Returns the number of slot variables modified.
        """
        if self.optimizer is None:
            return 0
        from ..diagnostics.surgery import project_out_conflict

        modified = 0
        for var in getattr(self.optimizer, "variables", []):
            path = getattr(var, "path", getattr(var, "name", "")) or ""
            if not ("momentum" in path or path.endswith("/m") or "_m" in path):
                continue
            g = _match_grad(path, task_grads_by_var)
            if g is None:
                continue
            g_cast = tf.cast(g, var.dtype)
            corrected, changed = project_out_conflict(
                tf.identity(var), g_cast, only_if_conflicting=True)
            if changed:
                var.assign(corrected)
                modified += 1
        return modified


class ConstraintTape:
    """Context manager wrapping a ``tf.GradientTape`` (Section 10.3)."""

    def __init__(self, controller=None, assertions=None, adapter: "TFAdapter" = None,
                 persistent: bool = True):
        _require()
        self.controller = controller
        self.assertions = list(assertions or [])
        self.adapter = adapter
        # A persistent tape is required: steer_gradients differentiates two
        # separate losses (task and constraint), which needs two gradient passes.
        self._tape = tf.GradientTape(persistent=True)
        self._constraint_loss = None

    def __enter__(self):
        self._tape.__enter__()
        return self

    def __exit__(self, *exc):
        # Compute the joint constraint loss WHILE the tape is still recording, so
        # its penalty ops are traced. If it were computed later (in
        # steer_gradients, after the context closes) the ops would not be
        # recorded and tape.gradient(constraint_loss, ...) would silently return
        # None -- the constraint would appear active but never actually steer.
        if exc[0] is None and self.assertions:
            try:
                self._constraint_loss = self._joint_constraint_loss()
            except Exception:
                self._constraint_loss = None
        return self._tape.__exit__(*exc)

    def gradient(self, target, sources, **kwargs):
        return self._tape.gradient(target, sources, **kwargs)

    def steer_gradients(self, primary_loss, variables):
        """Compute task gradients, apply gradient surgery against the joint
        constraint gradient (5.2), and return corrected gradients ready for
        ``optimizer.apply_gradients``. Must be called after the ``with`` block;
        the constraint loss was captured inside it (see ``__exit__``)."""
        from ..diagnostics.surgery import apply_gradient_surgery

        task_grads = self._tape.gradient(primary_loss, variables)
        if self._constraint_loss is None:
            return task_grads
        c_grads = self._tape.gradient(self._constraint_loss, variables)
        out = []
        for tg, cg in zip(task_grads, c_grads):
            if tg is None:
                out.append(None)
                continue
            if cg is None:
                out.append(tg)
                continue
            res = apply_gradient_surgery(tg, cg)
            out.append(tg + res.corrected_constraint_grad)
        return out

    def _joint_constraint_loss(self):
        total = 0.0
        for a in self.assertions:
            w = self.controller.states[a.name].weight if self.controller else 1.0
            total = total + w * a.violation()
        return total


class CompelloKerasCallback:
    """A ``keras.callbacks.Callback`` mirroring the HF/Lightning adapters (3.3).

    Works against any Keras 3 backend (torch/tensorflow/jax per KERAS_BACKEND)
    via ``keras.ops`` / ``compello.math``.
    """

    def __init__(self, controller, assertions, adapter: "TFAdapter"):
        _require()
        import keras  # noqa: F401

        self.controller = controller
        self.assertions = list(assertions)
        self.adapter = adapter
        self.controller.register_assertions(self.assertions)

    def on_train_batch_end(self, batch, logs=None):  # pragma: no cover
        local = {a.name: a.violation_scalar() for a in self.assertions}
        self.controller.step(self.adapter.sync_violations(local))


def _match_grad(var_path: str, task_grads_by_var: Dict[str, Any]):
    """Best-effort match of a Keras slot-variable path to a task gradient by
    name substring (the variable path embeds the owning layer/weight name)."""
    for key, grad in task_grads_by_var.items():
        if key in var_path or var_path in key:
            return grad
    return None


def register() -> None:
    if _HAS_TF:
        from .. import math as cmath

        cmath.register_backend(TFBackend())


register()
