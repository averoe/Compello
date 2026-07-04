"""compello.wrap / compello.unwrap -- the declaration model (Section 3.0).

``wrap()`` returns a *transparent proxy*, never a subclass and never a
monkeypatch of the underlying framework object (Section 9 "no monkeypatching").

- Wrapping a model returns a ``ModelProxy`` that forwards every attribute and
  call through to the underlying module, and additionally snapshots the most
  recent forward output so ``proxy.output`` yields a live ``OutputTarget``.
- Wrapping a raw tensor returns a thin ``TensorProxy`` whose ``__getitem__``
  yields a ``LogitTarget`` instead of an ordinary slice.

The output snapshot here is captured from the *return value* of the forward
call, so it fires after the (possibly compiled) forward completes and needs no
special ``torch.compile`` handling (Section 3.0 note). Backend adapters may
additionally register a native forward hook (Section 10); the portable core
relies only on the return-value capture, which works on every backend.
"""

from __future__ import annotations

from typing import Any

from . import math as cmath
from .targets import LogitTarget, OutputTarget

# Attribute names owned by the proxy itself (not forwarded to the wrapped obj).
_MODEL_RESERVED = {"_compello_wrapped", "_compello_last_output", "output", "unwrap"}


class _Proxy:
    """Marker base class so the rest of the library can recognise proxies."""

    __slots__ = ()


class ModelProxy(_Proxy):
    """Transparent proxy around a callable model."""

    def __init__(self, wrapped: Any):
        object.__setattr__(self, "_compello_wrapped", wrapped)
        object.__setattr__(self, "_compello_last_output", None)

    # -- transparent passthrough ----------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; forward to the wrapped module.
        return getattr(object.__getattribute__(self, "_compello_wrapped"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _MODEL_RESERVED:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_compello_wrapped"), name, value)

    def __call__(self, *args, **kwargs) -> Any:
        wrapped = object.__getattribute__(self, "_compello_wrapped")
        out = wrapped(*args, **kwargs)
        object.__setattr__(self, "_compello_last_output", out)
        return out

    def __repr__(self) -> str:
        wrapped = object.__getattribute__(self, "_compello_wrapped")
        return f"ModelProxy({wrapped!r})"

    # -- Compello surface ------------------------------------------------
    @property
    def output(self) -> OutputTarget:
        # Deferred, live target: reads the most recent forward output at
        # evaluation time (Section 3.0), so assertions declared before the
        # first forward still work and always see the current step.
        return OutputTarget(source=self)

    def current_output(self) -> Any:
        return object.__getattribute__(self, "_compello_last_output")

    def unwrap(self) -> Any:
        return object.__getattribute__(self, "_compello_wrapped")


class TensorProxy(_Proxy):
    """Thin indexable proxy around a raw tensor."""

    __slots__ = ("_compello_tensor",)

    def __init__(self, tensor: Any):
        object.__setattr__(self, "_compello_tensor", tensor)

    def __getitem__(self, index: Any) -> LogitTarget:
        return LogitTarget(self._compello_tensor, vocab_index=index)

    def as_output(self) -> OutputTarget:
        """Treat the wrapped tensor as a plain output (range/inequality)."""
        return OutputTarget(self._compello_tensor)

    @property
    def tensor(self) -> Any:
        return self._compello_tensor

    def unwrap(self) -> Any:
        return self._compello_tensor

    def __repr__(self) -> str:
        return f"TensorProxy(shape={_safe_shape(self._compello_tensor)})"


def _looks_like_tensor(obj: Any) -> bool:
    # Native to a registered backend?
    for be in cmath._REGISTRY.values():
        try:
            if be.is_native(obj):
                return True
        except Exception:
            continue
    # Array-like but not callable: has shape/__getitem__ and is not callable.
    if callable(obj):
        return False
    return hasattr(obj, "shape") or hasattr(obj, "__getitem__")


def wrap(obj: Any) -> Any:
    """Wrap a model or tensor, returning a transparent proxy (Section 3.0)."""
    if isinstance(obj, _Proxy):
        return obj
    if _looks_like_tensor(obj):
        return TensorProxy(obj)
    return ModelProxy(obj)


def unwrap(obj: Any) -> Any:
    """Return the underlying model/tensor from a proxy (inverse of wrap)."""
    if isinstance(obj, (ModelProxy, TensorProxy)):
        return obj.unwrap()
    return obj


def _safe_shape(t: Any):
    s = getattr(t, "shape", None)
    if s is None:
        return None
    try:
        return tuple(s)
    except TypeError:
        return s
