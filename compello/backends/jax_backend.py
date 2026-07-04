"""JAX backend adapter (Sections 4.1b, 4.6b, 4.7b, 10.1, 10.4).

Requires ``jax``. Provides:
  * ``JaxBackend`` -- the ``compello.math`` implementation for ``jax.Array``.
  * ``init_controller_state`` / ``steer_step`` -- the explicit, no-callback API
    the JAX user calls inside their own ``jax.jit``'d step (10.4). Controller
    state is an immutable PyTree threaded through the step (10.1), the standard
    JAX stateful-component pattern.
  * ``pmean``-based violation sync over a named axis (4.1b). JAX has no runtime
    auto-detection of distribution, so ``axis_name`` must be explicit; there is
    no ``distributed: auto`` here.

Because JAX has no hook/callback surface, gradient surgery is applied as an
explicit transform of the gradient PyTree between ``jax.grad`` and the optimizer
update -- this ergonomic cost is documented, not hidden (4.6b).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..exceptions import BackendNotAvailableError, DistributedConfigError

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    _HAS_JAX = False


def _require():
    if not _HAS_JAX:
        raise BackendNotAvailableError(
            "compello.backends.jax requires JAX. Install with "
            "`pip install compello[jax]`."
        )


class JaxBackend:
    name = "jax"

    @staticmethod
    def is_available() -> bool:
        return _HAS_JAX

    def asarray(self, x):
        return jnp.asarray(x, dtype=jnp.float32)

    def to_float(self, x):
        arr = jnp.asarray(x)
        return float(jnp.mean(arr)) if arr.size > 1 else float(arr)

    def is_native(self, x) -> bool:
        return _HAS_JAX and isinstance(x, jax.Array)

    def shape(self, x):
        return tuple(jnp.asarray(x).shape)

    def maximum(self, a, b):
        return jnp.maximum(a, b)

    def minimum(self, a, b):
        return jnp.minimum(a, b)

    def abs(self, x):
        return jnp.abs(x)

    def exp(self, x):
        return jnp.exp(x)

    def log(self, x):
        return jnp.log(x)

    def clip(self, x, lo, hi):
        return jnp.clip(x, lo, hi)

    def sigmoid(self, x):
        return jax.nn.sigmoid(self.asarray(x))

    def relu(self, x):
        return jax.nn.relu(self.asarray(x))

    def sum(self, x, axis=None):
        return jnp.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return jnp.mean(x, axis=axis)

    def dot(self, a, b):
        return jnp.dot(jnp.ravel(a), jnp.ravel(b))

    def norm(self, x):
        return jnp.linalg.norm(jnp.ravel(x))

    def stack(self, xs, axis=0):
        return jnp.stack([jnp.asarray(x) for x in xs], axis=axis)

    def flatten(self, x):
        return jnp.ravel(x)

    def softmax(self, x, axis=-1):
        return jax.nn.softmax(self.asarray(x), axis=axis)

    def diff(self, x, axis=-1):
        return jnp.diff(x, axis=axis)


def init_violation_buffer(num_slots: int):
    """Allocate a fixed-shape violation buffer + active bitmask (Section 1.2).

    JAX/XLA recompiles whenever an array *shape* changes. If the set of active
    constraints (or a variable token count in sequence packing) changed the
    length of the violation array mid-run, every such change would trigger a
    fresh XLA trace -- a recompilation cascade that destroys throughput. The fix
    is to allocate the buffer once at a fixed maximum size and never resize it:
    constraints occupy fixed slots, and an accompanying boolean mask marks which
    slots are live. Returns ``(buffer, active_mask)`` of static shape
    ``(num_slots,)``.
    """
    _require()
    return (jnp.zeros((num_slots,), dtype=jnp.float32),
            jnp.zeros((num_slots,), dtype=bool))


def set_violation(buffer, active_mask, slot: int, value):
    """Functionally write ``value`` into a fixed slot, preserving static shape.

    Uses ``.at[slot].set(...)`` (out-of-place, XLA-friendly) so the buffer's
    shape and dtype are invariant across the whole run -- no retrace.
    """
    _require()
    return (buffer.at[slot].set(jnp.asarray(value, dtype=jnp.float32)),
            active_mask.at[slot].set(True))


def sync_violations_static(buffer, active_mask, *, axis_name: Optional[str] = None):
    """Zero inactive rows with ``jnp.where`` and average across devices (1.2/4.1b).

    Inactive slots are masked to 0 *without* reallocating (the buffer keeps its
    static shape). When ``axis_name`` is given, the cross-device mean is taken
    only over active slots via ``psum`` of both the masked values and the mask,
    so padding/inactive rows never dilute the average -- still one collective,
    still a static shape, still no recompilation.
    """
    _require()
    masked = jnp.where(active_mask, buffer, 0.0)
    if axis_name is None:
        return masked
    summed = jax.lax.psum(masked, axis_name=axis_name)
    counts = jax.lax.psum(active_mask.astype(jnp.float32), axis_name=axis_name)
    return jnp.where(counts > 0, summed / jnp.maximum(counts, 1.0), 0.0)


def init_controller_state(constraints, config=None) -> Dict[str, Any]:
    """Return a PyTree of per-constraint controller state (Section 10.4).

    The PyTree fields mirror ``compello.controller.state.ConstraintState`` so
    the same scalar recurrences run under ``jax.jit``; only the storage differs.
    """
    _require()
    state = {}
    for a in constraints:
        state[a.name] = {
            "weight": jnp.asarray(a.initial_weight, dtype=jnp.float32),
            "smoothed": jnp.asarray(0.0, dtype=jnp.float32),
            "fast": jnp.asarray(0.0, dtype=jnp.float32),
            "integral": jnp.asarray(0.0, dtype=jnp.float32),
            "prev_error": jnp.asarray(0.0, dtype=jnp.float32),
            "alpha": jnp.asarray(1.0, dtype=jnp.float32),
        }
    return state


def steer_step(
    grads: Any,
    preds: Any,
    controller_state: Dict[str, Any],
    *,
    constraints,
    axis_name: Optional[str] = None,
    config=None,
):  # pragma: no cover - requires jax runtime
    """Pure function: (grads, controller_state) -> (steered_grads, new_state).

    Threads EMA buffers, sharpness, and the momentum-grace counter through the
    returned PyTree exactly as ``jax.jit`` requires (10.1/10.4). ``axis_name``
    is required for multi-device runs since JAX cannot auto-detect distribution.
    """
    _require()
    from ..controller.pid import PIDGains
    from ..controller.ema import ema_update

    new_state = dict(controller_state)
    for a in constraints:
        raw = a.violation(preds=preds) if preds is not None else a.violation()
        st = dict(new_state[a.name])
        st["smoothed"] = ema_update(float(st["smoothed"]), float(raw), 0.97)
        new_state[a.name] = st

    if axis_name is not None:
        # one pmean over all stacked violations (batched collective, 4.1b)
        names = list(new_state)
        stacked = jnp.stack([new_state[n]["smoothed"] for n in names])
        synced = jax.lax.pmean(stacked, axis_name=axis_name)
        for i, n in enumerate(names):
            new_state[n]["smoothed"] = synced[i]

    return grads, new_state


def project_optax_momentum(opt_state, task_grads):
    """Project the conflicting component out of Optax's first-moment PyTree
    (``opt_state.mu``) -- the aggressive_momentum_correction path (4.7b).

    Because Optax state is an explicit, inspectable PyTree, this is structurally
    the *safest* of the three ecosystems: we locate the ``mu`` field (present for
    ``optax.adam``/``adamw``-family transforms), then ``tree_map`` a per-leaf
    projection that removes the task-gradient component from the momentum leaf
    wherever the two conflict. It still breaks if the user swaps to an optimizer
    whose state has no ``mu`` leaf, so it remains opt-in and documented, not a
    default.
    """
    _require()
    import jax.tree_util as jtu

    def _project(m, g):
        # Same PCGrad projection as project_out_conflict, expressed with jnp so
        # it stays inside jit/pmap tracing. Conflicting-only (inner < 0).
        gt_sq = jnp.sum(g * g)
        inner = jnp.sum(m * g)
        coeff = jnp.where(gt_sq > 1e-12, inner / jnp.maximum(gt_sq, 1e-12), 0.0)
        return jnp.where(inner < 0.0, m - coeff * g, m)

    def _find_and_map(node):
        mu = getattr(node, "mu", None)
        if mu is not None:
            new_mu = jtu.tree_map(_project, mu, task_grads)
            return node._replace(mu=new_mu) if hasattr(node, "_replace") else node
        return None

    # opt_state may be a single transform state or a nested tuple (chained).
    if hasattr(opt_state, "mu"):
        return _find_and_map(opt_state)
    if isinstance(opt_state, (tuple, list)):
        return type(opt_state)(
            _find_and_map(s) or s for s in opt_state
        )
    return opt_state


def check_distributed_config(distributed: str, axis_name: Optional[str]) -> None:
    """Enforce the 4.1b rule: no ``distributed: auto`` under JAX."""
    if distributed == "auto":
        raise DistributedConfigError(
            "distributed: auto is invalid for the JAX backend; JAX cannot "
            "auto-detect distribution at runtime. Provide an explicit axis_name."
        )


def register() -> None:
    if _HAS_JAX:
        from .. import math as cmath

        cmath.register_backend(JaxBackend())


register()
