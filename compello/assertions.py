"""expect() and the Assertion object (Sections 3.1 / 3.1a).

``expect()`` decides *which* penalty to apply by dispatching on the type of the
(already typed) target, then on which keyword argument was supplied -- never by
introspecting a predicate lambda's body. If the target has no typed wrapper and
no explicit ``assertion_type`` was given, ``AmbiguousAssertionError`` is raised
at declaration time.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import math as cmath
from . import penalties as P
from .exceptions import AmbiguousAssertionError
from .proxy import ModelProxy, TensorProxy, unwrap
from .targets import LogitTarget, ModelTarget, OutputTarget, RawTarget

# assertion kinds
RANGE = "range"
MONOTONICITY = "monotonicity"
INVARIANCE = "invariance"
PROBABILITY_FLOOR = "probability_floor"
PARITY = "parity"
LIPSCHITZ = "lipschitz"
CONSISTENCY = "consistency"
PROXY = "proxy"

_KEYWORD_KINDS = {
    "monotonic_in": MONOTONICITY,
    "invariant_to": INVARIANCE,
    "parity_across": PARITY,
    "lipschitz_bound": LIPSCHITZ,
    "consistent_across": CONSISTENCY,
}

_counter = {"n": 0}

# Custom assertion types (Section 3.1: "users can register custom assertion
# types by supplying their own violation function and penalty shape").
# Maps a user-chosen name -> a callable(tensor, **params) -> backend-native
# non-negative penalty scalar.
_CUSTOM_TYPES: Dict[str, Callable[..., Any]] = {}


def register_assertion_type(name: str, violation_fn: Callable[..., Any]) -> None:
    """Register a custom assertion type usable via ``assertion_type=name``.

    ``violation_fn(tensor, **params)`` must return a non-negative, differentiable
    penalty (0 when satisfied), expressed against ``compello.math`` so it runs on
    any backend. It is invoked with the resolved target tensor and any extra
    keyword args passed to ``expect``.
    """
    if not callable(violation_fn):
        raise TypeError("violation_fn must be callable")
    if name in {RANGE, MONOTONICITY, INVARIANCE, PROBABILITY_FLOOR, PARITY,
                LIPSCHITZ, CONSISTENCY, PROXY}:
        raise ValueError(f"{name!r} is a built-in assertion type; choose another name")
    _CUSTOM_TYPES[name] = violation_fn


def registered_assertion_types() -> List[str]:
    return sorted(_CUSTOM_TYPES)


class Assertion:
    """A declared correctness property with a differentiable penalty.

    ``violation(**inputs)`` computes the current penalty scalar. Inputs may be
    supplied explicitly (used by tests and by backend adapters that materialise
    tensors each step) or resolved lazily from the stored target.
    """

    def __init__(
        self,
        kind: str,
        *,
        name: str,
        resolver: Optional[Callable[[], Any]] = None,
        op: Optional[str] = None,
        threshold: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        initial_weight: float = 1.0,
        respect_loss_mask: bool = False,
        target_id: Any = None,
    ):
        self.kind = kind
        self.name = name
        self.resolver = resolver
        self.op = op
        self.threshold = threshold
        self.params = params or {}
        self.initial_weight = float(initial_weight)
        self.respect_loss_mask = respect_loss_mask
        # Best-effort identity of the assertion's target, used by pre-flight
        # conflict detection (6.1) to group assertions on the same target.
        self.target_id = target_id

    # -- evaluation ------------------------------------------------------
    def _primary(self, override: Any) -> Any:
        if override is not None:
            return override
        if self.resolver is not None:
            return self.resolver()
        raise ValueError(
            f"assertion {self.name!r} has no resolvable target; pass the tensor "
            f"explicitly to violation()."
        )

    def violation(self, tensor: Any = None, **inputs: Any) -> Any:
        """Return the (backend-native) penalty scalar for the current state."""
        k = self.kind
        if k == RANGE:
            return P.hinge_range(self._primary(tensor), self.op, self.threshold)
        if k == MONOTONICITY:
            return P.monotonicity(
                self._primary(tensor),
                increasing=self.params.get("increasing", True),
                axis=self.params.get("axis", -1),
            )
        if k == PROBABILITY_FLOOR:
            mask = inputs.get("mask") if self.respect_loss_mask else None
            return P.probability_floor(
                self._primary(tensor),
                self.params["index"],
                self.op,
                self.threshold,
                mask=mask,
            )
        if k == INVARIANCE:
            return P.invariance_l2(inputs["output_a"], inputs["output_b"])
        if k == CONSISTENCY:
            return P.consistency(inputs["rep_a"], inputs["rep_b"])
        if k == PARITY:
            return P.parity(inputs["rates_group_a"], inputs["rates_group_b"])
        if k == LIPSCHITZ:
            return P.lipschitz(inputs["grad_norm"], self.params["bound"])
        if k == PROXY:
            return P.sigmoid_relaxation(inputs["margin"], self.params.get("alpha", 1.0))
        if k in _CUSTOM_TYPES:
            fn = _CUSTOM_TYPES[k]
            params = {kk: vv for kk, vv in self.params.items()}
            params.update(inputs)
            return fn(self._primary(tensor), **params)
        raise ValueError(f"unknown assertion kind {k!r}")

    def violation_scalar(self, tensor: Any = None, **inputs: Any) -> float:
        return cmath.to_float(self.violation(tensor=tensor, **inputs))

    def __repr__(self) -> str:
        cond = f" {self.op} {self.threshold}" if self.op else ""
        return f"Assertion({self.name!r}, kind={self.kind}{cond})"


def _auto_name(kind: str) -> str:
    _counter["n"] += 1
    return f"{kind}_{_counter['n']}"


def expect(
    target: Any,
    predicate: Any = None,
    *,
    assertion_type: Optional[str] = None,
    name: Optional[str] = None,
    initial_weight: float = 1.0,
    respect_loss_mask: bool = False,
    monotonic_in: Optional[str] = None,
    invariant_to: Optional[Callable[[Any], Any]] = None,
    parity_across: Optional[str] = None,
    lipschitz_bound: Optional[float] = None,
    consistent_across: Optional[List[str]] = None,
    increasing: bool = True,
    alpha: float = 1.0,
    **extra: Any,
) -> Assertion:
    """Declare an assertion. See Section 3.1 for the assertion catalogue.

    Extra keyword arguments are forwarded as parameters to a custom assertion
    type's violation function (registered via ``register_assertion_type``).
    """

    # 1) keyword-driven assertions dispatch on the keyword name (unambiguous).
    supplied_kw = {
        "monotonic_in": monotonic_in,
        "invariant_to": invariant_to,
        "parity_across": parity_across,
        "lipschitz_bound": lipschitz_bound,
        "consistent_across": consistent_across,
    }
    active_kw = {k: v for k, v in supplied_kw.items() if v is not None}
    if len(active_kw) > 1:
        raise ValueError(f"only one keyword assertion may be given, got {list(active_kw)}")

    if active_kw:
        (kw, val), = active_kw.items()
        kind = _KEYWORD_KINDS[kw]
        params: Dict[str, Any] = {}
        if kw == "monotonic_in":
            params = {"feature": val, "increasing": increasing}
        elif kw == "invariant_to":
            params = {"transform": val}
        elif kw == "parity_across":
            params = {"attribute": val}
        elif kw == "lipschitz_bound":
            params = {"bound": float(val)}
        elif kw == "consistent_across":
            params = {"views": list(val)}
        underlying = unwrap(target) if isinstance(target, (ModelProxy, TensorProxy)) else target
        return Assertion(
            kind, name=name or _auto_name(kind), params=params,
            initial_weight=initial_weight, target_id=id(underlying),
        )

    # 2) type-driven dispatch for predicate assertions.
    tgt, resolver, target_id = _normalise_target(target)

    if isinstance(tgt, OutputTarget):
        if assertion_type in (None, "range"):
            op, thr = P.parse_condition(predicate)
            return Assertion(
                RANGE, name=name or _auto_name(RANGE), resolver=resolver,
                op=op, threshold=thr, initial_weight=initial_weight,
                target_id=target_id,
            )
        if assertion_type == "proxy":
            return Assertion(
                PROXY, name=name or _auto_name(PROXY), resolver=resolver,
                params={"alpha": alpha}, initial_weight=initial_weight,
                target_id=target_id,
            )
        if assertion_type in _CUSTOM_TYPES:
            return Assertion(
                assertion_type, name=name or _auto_name(assertion_type),
                resolver=resolver, initial_weight=initial_weight,
                target_id=target_id, params=dict(extra),
            )
        raise ValueError(f"assertion_type {assertion_type!r} invalid for OutputTarget")

    if isinstance(tgt, LogitTarget):
        op, thr = P.parse_condition(predicate)
        return Assertion(
            PROBABILITY_FLOOR, name=name or _auto_name(PROBABILITY_FLOOR),
            resolver=resolver, op=op, threshold=thr,
            params={"index": tgt.vocab_index}, initial_weight=initial_weight,
            respect_loss_mask=respect_loss_mask,
            target_id=target_id,
        )

    if isinstance(tgt, ModelTarget):
        raise AmbiguousAssertionError(
            "a bare model target requires a keyword assertion (invariant_to=, "
            "parity_across=, lipschitz_bound=, monotonic_in=, consistent_across=)."
        )

    # 3) raw target -- escape hatch requires an explicit assertion_type.
    if assertion_type is None:
        raise AmbiguousAssertionError(
            "target has no typed wrapper and no assertion_type was given. Wrap it "
            "with compello.wrap(...) or pass assertion_type='range' (Section 3.1a)."
        )
    if assertion_type == "range":
        op, thr = P.parse_condition(predicate)
        return Assertion(
            RANGE, name=name or _auto_name(RANGE),
            resolver=lambda: tgt.tensor, op=op, threshold=thr,
            initial_weight=initial_weight, target_id=id(tgt.tensor),
        )
    if assertion_type == "proxy":
        return Assertion(
            PROXY, name=name or _auto_name(PROXY), params={"alpha": alpha},
            initial_weight=initial_weight,
        )
    if assertion_type in _CUSTOM_TYPES:
        return Assertion(
            assertion_type, name=name or _auto_name(assertion_type),
            resolver=(lambda: tgt.tensor), initial_weight=initial_weight,
            target_id=id(tgt.tensor), params=dict(extra),
        )
    raise ValueError(f"unknown assertion_type {assertion_type!r}")


def _normalise_target(target: Any):
    """Return ``(typed_target, resolver, target_id)`` for the user target."""
    if isinstance(target, ModelProxy):
        return ModelTarget(target), (lambda: target.output.tensor), id(unwrap(target))
    if isinstance(target, TensorProxy):
        ot = target.as_output()
        return ot, (lambda: target.tensor), id(target.tensor)
    if isinstance(target, OutputTarget):
        snap = target
        tid = id(snap.source) if snap.source is not None else id(snap._tensor)
        return snap, (lambda: snap.tensor), tid
    if isinstance(target, LogitTarget):
        snap = target
        return snap, (lambda: snap.tensor), ("logit", id(snap.tensor), snap.vocab_index)
    if isinstance(target, ModelTarget):
        return target, None, id(target.module)
    # unwrapped raw value
    return RawTarget(target), None, id(target)
