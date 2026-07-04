"""Typed assertion targets (Section 3.0 / 3.1a).

``expect()`` dispatches on the *type* of its target, not on the body of a
predicate lambda (which cannot be reliably introspected). These typed wrappers
are what ``compello.wrap(...)`` produces, and they carry just enough metadata
for dispatch to be unambiguous.
"""

from __future__ import annotations

from typing import Any, Optional


class Target:
    """Base class for all typed targets."""

    __slots__ = ()


class OutputTarget(Target):
    """A model output tensor (Section 3.1a).

    Produced by ``wrapped_model.output``. A bare predicate on this target
    dispatches to a range/inequality (hinge) penalty.
    """

    __slots__ = ("_tensor", "value_range", "source")

    def __init__(self, tensor: Any = None, value_range: Optional[tuple] = None, source: Any = None):
        # ``source`` is the ModelProxy this target reads from, when the target
        # is deferred (declared before/independent of a specific snapshot). When
        # present, ``.tensor`` reads the proxy's most recent forward output live,
        # so an assertion tracks the current step rather than a stale snapshot.
        self._tensor = tensor
        self.value_range = value_range
        self.source = source

    @property
    def tensor(self) -> Any:
        if self.source is not None:
            live = self.source.current_output()
            if live is None:
                raise RuntimeError(
                    "output target evaluated before any forward pass ran on the "
                    "wrapped model."
                )
            return live
        return self._tensor

    def __repr__(self) -> str:
        try:
            shp = _safe_shape(self.tensor)
        except RuntimeError:
            shp = "deferred"
        return f"OutputTarget(shape={shp}, value_range={self.value_range})"


class LogitTarget(Target):
    """An indexed logit / probability target (Section 3.1a).

    Produced by ``compello.wrap(logits)[token]``. A bare predicate dispatches to
    a probability-floor penalty; the predicate is evaluated in probability space
    (after an internal softmax/sigmoid), never on raw logits directly.
    """

    __slots__ = ("tensor", "vocab_index", "is_probability_space")

    def __init__(self, tensor: Any, vocab_index: Any, is_probability_space: bool = False):
        self.tensor = tensor
        self.vocab_index = vocab_index
        self.is_probability_space = is_probability_space

    def __repr__(self) -> str:
        return f"LogitTarget(vocab_index={self.vocab_index}, prob_space={self.is_probability_space})"


class ModelTarget(Target):
    """The model itself (Section 3.1a).

    Produced by passing a wrapped model directly. Used only with keyword
    assertions: ``invariant_to=``, ``parity_across=``, ``lipschitz_bound=``,
    ``consistent_across=``, ``monotonic_in=``.
    """

    __slots__ = ("module",)

    def __init__(self, module: Any):
        self.module = module

    def __repr__(self) -> str:
        return f"ModelTarget({type(self.module).__name__})"


class RawTarget(Target):
    """A raw value with no typed wrapper (Section 3.1a escape hatch).

    Automatic dispatch cannot proceed for this target; an explicit
    ``assertion_type=`` keyword is required or ``AmbiguousAssertionError`` is
    raised at declaration time.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: Any):
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"RawTarget(shape={_safe_shape(self.tensor)})"


def _safe_shape(t: Any):
    for attr in ("shape",):
        s = getattr(t, attr, None)
        if s is not None:
            try:
                return tuple(s)
            except TypeError:
                return s
    return None
