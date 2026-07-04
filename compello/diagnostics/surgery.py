"""Gradient-surgery conflict resolution (Section 5.2).

Computes the cosine similarity between the primary task gradient and the joint
constraint gradient. When they point in substantially opposing directions the
conflicting component is projected out of the constraint gradient before the
update -- the PCGrad projection of Yu et al., applied here specifically between
a task loss and a live, adaptively-weighted constraint gradient.

Written against ``compello.math`` so it runs on any backend's gradient tensors.
``gradient_surgery_scope`` (Section 5.6) is applied by the caller, which passes
only the in-scope slice of the gradient here -- this function does not itself
walk the model; it operates on whatever vectors it is given, keeping the O(N)
cost budgeted by the caller (5.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .. import math as cmath

# Above this parameter count, full-model gradient surgery is expensive enough to
# warrant a pre-flight warning (Section 5.6).
FULL_MODEL_PARAM_WARN_THRESHOLD = 1_000_000_000


@dataclass
class SurgeryResult:
    corrected_constraint_grad: Any
    cosine_similarity: float
    conflict: bool
    projected: bool


@dataclass
class ScopedSurgeryResult:
    corrected: Dict[str, Any]     # per-parameter corrected constraint gradients
    cosine_similarity: float
    conflict: bool
    projected: bool
    params_in_scope: int
    params_total: int

    @property
    def cost_fraction(self) -> float:
        """Fraction of the full-model gradient the surgery actually touched."""
        return self.params_in_scope / max(1, self.params_total)


def project_out_conflict(vector: Any, direction: Any, *, only_if_conflicting: bool = True):
    """Remove ``direction``'s component from ``vector`` (the PCGrad projection).

    Backend-agnostic (``compello.math``), so the *same* tested code path runs on
    numpy, torch, tf, or jax tensors. This is the shared kernel behind both
    task/constraint gradient surgery (5.2) and the aggressive momentum-buffer
    correction (4.7b) -- the adapters only read/write framework buffers and
    delegate the math here.

    Returns ``(corrected_vector, changed)``. When ``only_if_conflicting`` (the
    momentum-bleed case), the projection is applied only if ``vector`` points
    against ``direction`` (negative inner product); otherwise a non-conflicting
    momentum buffer is left untouched.
    """
    v = cmath.flatten(vector)
    d = cmath.flatten(direction)
    d_sq = cmath.to_float(cmath.dot(d, d))
    if d_sq <= 1e-12:
        return vector, False
    inner = cmath.to_float(cmath.dot(v, d))
    if only_if_conflicting and inner >= 0.0:
        return vector, False
    coeff = inner / d_sq
    return vector - coeff * direction, True


def cosine_similarity(a: Any, b: Any) -> float:
    fa = cmath.flatten(a)
    fb = cmath.flatten(b)
    denom = cmath.to_float(cmath.norm(fa)) * cmath.to_float(cmath.norm(fb))
    if denom <= 1e-12:
        return 0.0
    return cmath.to_float(cmath.dot(fa, fb)) / denom


def apply_gradient_surgery(
    task_grad: Any,
    constraint_grad: Any,
    *,
    conflict_threshold: float = 0.0,
) -> SurgeryResult:
    """Project the conflicting component out of ``constraint_grad``.

    A conflict is declared when cosine similarity < ``conflict_threshold``
    (default 0.0 -- i.e. pointing in opposing half-spaces). When it conflicts,
    subtract the projection of the constraint gradient onto the task gradient:
    ``g_c' = g_c - (g_c . g_t / g_t . g_t) g_t``.
    """
    cos = cosine_similarity(task_grad, constraint_grad)
    if cos >= conflict_threshold:
        return SurgeryResult(constraint_grad, cos, conflict=False, projected=False)

    gt = cmath.flatten(task_grad)
    gc = cmath.flatten(constraint_grad)
    gt_sq = cmath.to_float(cmath.dot(gt, gt))
    if gt_sq <= 1e-12:
        return SurgeryResult(constraint_grad, cos, conflict=True, projected=False)
    coeff = cmath.to_float(cmath.dot(gc, gt)) / gt_sq
    corrected = gc - coeff * gt
    return SurgeryResult(corrected, cos, conflict=True, projected=True)


# --------------------------------------------------------------------------
# Layer-scoped surgery over a per-parameter gradient dict (Section 5.6)
# --------------------------------------------------------------------------

def select_in_scope(names: List[str], scope: Optional[str]) -> Set[str]:
    """Resolve ``gradient_surgery_scope`` to the set of in-scope parameter names.

    Supports ``None``/``"full_model"`` (all), ``"last_n_layers:K"`` (the last K
    named parameters -- the layers nearest the output, where late-representation
    conflict concentrates, 5.6), and ``"modules:a,b"`` (any name containing one
    of the substrings).
    """
    if not scope or scope == "full_model":
        return set(names)
    if scope.startswith("last_n_layers:"):
        try:
            k = int(scope.split(":", 1)[1])
        except ValueError:
            return set(names)
        return set(names[-k:]) if k < len(names) else set(names)
    if scope.startswith("modules:"):
        wanted = [w.strip() for w in scope.split(":", 1)[1].split(",") if w.strip()]
        return {n for n in names if any(w in n for w in wanted)}
    return set(names)


def scoped_gradient_surgery(
    task_grads: Dict[str, Any],
    constraint_grads: Dict[str, Any],
    *,
    scope: Optional[str] = None,
    conflict_threshold: float = 0.0,
) -> ScopedSurgeryResult:
    """PCGrad projection restricted to an in-scope subset of parameters (5.6).

    The joint cosine and projection coefficient are computed by accumulating
    dot-products/norms across the in-scope parameters (dot and norm are additive
    over a partition), so this is mathematically identical to concatenating the
    in-scope gradients into one vector and projecting -- without ever allocating
    the concatenation. Out-of-scope gradients are returned unchanged, bounding
    the O(N) cost to the in-scope fraction.
    """
    names = [n for n in constraint_grads if n in task_grads]
    in_scope = select_in_scope(names, scope)

    dot_tc = 0.0
    norm_t2 = 0.0
    norm_c2 = 0.0
    for n in in_scope:
        tg = cmath.flatten(task_grads[n])
        cg = cmath.flatten(constraint_grads[n])
        dot_tc += cmath.to_float(cmath.dot(tg, cg))
        norm_t2 += cmath.to_float(cmath.dot(tg, tg))
        norm_c2 += cmath.to_float(cmath.dot(cg, cg))

    denom = (norm_t2 ** 0.5) * (norm_c2 ** 0.5)
    cos = dot_tc / denom if denom > 1e-12 else 0.0

    params_total = sum(_numel(constraint_grads[n]) for n in names)
    params_in_scope = sum(_numel(constraint_grads[n]) for n in in_scope)

    corrected: Dict[str, Any] = dict(constraint_grads)
    if cos >= conflict_threshold or norm_t2 <= 1e-12:
        return ScopedSurgeryResult(corrected, cos, conflict=cos < conflict_threshold,
                                   projected=False, params_in_scope=params_in_scope,
                                   params_total=params_total)

    coeff = dot_tc / norm_t2
    for n in in_scope:
        corrected[n] = constraint_grads[n] - coeff * task_grads[n]
    return ScopedSurgeryResult(corrected, cos, conflict=True, projected=True,
                               params_in_scope=params_in_scope, params_total=params_total)


def full_model_scope_warning(scope: Optional[str], param_count: int) -> Optional[str]:
    """Return a non-blocking warning if full-model surgery is used on a very
    large model (Section 5.6), else None."""
    if scope == "full_model" and param_count > FULL_MODEL_PARAM_WARN_THRESHOLD:
        return (
            f"gradient_surgery_scope=full_model on a {param_count/1e9:.1f}B-parameter "
            f"model runs an O(N) cosine/projection over every gradient each step. "
            f"Consider 'last_n_layers:K' to bound the cost (Section 5.6)."
        )
    return None


def _numel(x: Any) -> int:
    shape = cmath.shape(x)
    n = 1
    for d in shape:
        n *= int(d)
    return n
