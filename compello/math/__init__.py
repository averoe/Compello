"""compello.math -- the shared mathematical core (Section 9).

All penalty functions and array-level diagnostics are written *once* against
this dispatcher. At call time an operation is routed to the backend that owns
its argument (a ``torch.Tensor`` goes to the torch backend, a ``jax.Array`` to
the jax backend, an ``np.ndarray`` to the numpy reference backend). This is the
mechanism that lets one ``expect()`` call execute identically on any backend
without duplicating the math per framework.

The dispatcher itself has zero hard dependencies. Backends register themselves
lazily; only backends whose underlying library is importable become available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import BackendNotAvailableError
from ._base import Backend

_REGISTRY: Dict[str, Backend] = {}
_ACTIVE: Optional[str] = None

# Operations that dispatch on the backend owning the first array-like argument.
_ELEMENTWISE = (
    "maximum", "minimum", "abs", "exp", "log", "clip", "sigmoid", "relu",
    "sum", "mean", "dot", "norm", "stack", "flatten", "softmax", "diff",
    "asarray", "to_float", "shape",
)


def register_backend(backend: Backend, *, make_active: bool = False) -> None:
    """Register a backend instance. Only registers if it is available."""
    if not backend.is_available():
        return
    _REGISTRY[backend.name] = backend
    global _ACTIVE
    if make_active or _ACTIVE is None:
        _ACTIVE = backend.name


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def set_backend(name: str) -> None:
    global _ACTIVE
    if name not in _REGISTRY:
        raise BackendNotAvailableError(
            f"backend {name!r} is not registered/available; "
            f"available: {available_backends()}"
        )
    _ACTIVE = name


def get_backend(name: Optional[str] = None) -> Backend:
    if name is not None:
        if name not in _REGISTRY:
            raise BackendNotAvailableError(f"backend {name!r} is not available")
        return _REGISTRY[name]
    if _ACTIVE is None:
        raise BackendNotAvailableError(
            "no Compello backend is available. Install one of numpy/torch/"
            "tensorflow/jax, or register a backend via compello.math.register_backend()."
        )
    return _REGISTRY[_ACTIVE]


def backend_for(x: Any) -> Backend:
    """Return the backend that owns array ``x``, else the active backend."""
    for be in _REGISTRY.values():
        try:
            if be.is_native(x):
                return be
        except Exception:
            continue
    return get_backend()


def _make_dispatcher(op_name: str):
    def _op(*args, **kwargs):
        # find the first array-like positional arg to decide the backend
        target = args[0] if args else None
        be = backend_for(target)
        return getattr(be, op_name)(*args, **kwargs)

    _op.__name__ = op_name
    return _op


# Expose each op at module level, dispatching by argument backend.
for _name in _ELEMENTWISE:
    globals()[_name] = _make_dispatcher(_name)


def _register_default_backends() -> None:
    """Register the numpy reference backend if numpy is importable, and the
    keras.ops backend if Keras 3 is present (available, not auto-active)."""
    try:
        from ._numpy_backend import NumpyBackend

        register_backend(NumpyBackend())
    except Exception:  # pragma: no cover
        pass
    try:
        from ._keras_backend import KerasOpsBackend

        register_backend(KerasOpsBackend())  # available; not made active
    except Exception:  # pragma: no cover
        pass


_register_default_backends()

__all__ = [
    "Backend",
    "register_backend",
    "available_backends",
    "set_backend",
    "get_backend",
    "backend_for",
    *_ELEMENTWISE,
]
