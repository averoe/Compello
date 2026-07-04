"""Differentiable relaxations of non-differentiable metrics (Section 5.1).

Each modality in the plan has a hard, non-differentiable metric that must be
relaxed into something with a usable gradient before a constraint on it can be
steered. These are the concrete relaxations behind the 5.1 table:

  Tabular   -> soft F1 / soft confusion-rate (macro continuous relaxation)
  Vision    -> soft IoU (continuous intersection-over-union)
  Audio     -> spectral/amplitude gate (smooth log relaxation of a hard gate)
  Multimodal-> soft top-k rank (relaxation from strict rank to cosine-distance)

All are written against ``compello.math`` so they run on any backend, and all
return a non-negative *penalty* (0 when the target is met) suitable for the
controller. They can be attached as custom assertion types via
``register_assertion_type`` or used directly.
"""

from __future__ import annotations

from typing import Any

from . import math as cmath


def soft_iou_penalty(pred_mask: Any, true_mask: Any, *, target: float = 0.7) -> Any:
    """Continuous IoU relaxation (Vision, 5.1).

    ``pred_mask`` values are treated as soft occupancies in [0, 1] (apply a
    sigmoid upstream if they are logits). Soft intersection = sum(p*t), soft
    union = sum(p + t - p*t). Penalty = relu(target - softIoU).
    """
    p = cmath.asarray(pred_mask)
    t = cmath.asarray(true_mask)
    inter = cmath.sum(p * t)
    union = cmath.sum(p + t - p * t)
    iou = cmath.to_float(inter) / (cmath.to_float(union) + 1e-12)
    return cmath.relu(cmath.asarray(target - iou))


def soft_f1_penalty(probs: Any, labels: Any, *, target: float = 0.8) -> Any:
    """Macro soft-F1 relaxation (Tabular, 5.1).

    Uses probabilistic counts: tp = sum(p*y), fp = sum(p*(1-y)),
    fn = sum((1-p)*y). Soft-F1 = 2tp / (2tp + fp + fn). Penalty = relu(target - F1).
    """
    p = cmath.asarray(probs)
    y = cmath.asarray(labels)
    tp = cmath.to_float(cmath.sum(p * y))
    fp = cmath.to_float(cmath.sum(p * (1.0 - y)))
    fn = cmath.to_float(cmath.sum((1.0 - p) * y))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
    return cmath.relu(cmath.asarray(target - f1))


def spectral_gate_penalty(amplitude: Any, *, floor_db: float = -60.0, ref: float = 1.0) -> Any:
    """Smooth log relaxation of a hard amplitude gate (Audio, 5.1).

    Converts amplitude to dB (20*log10(|a|/ref)) and penalises energy that falls
    below ``floor_db`` -- i.e. relu(floor_db - level_db), averaged. A hard gate
    ("amplitude must exceed X or be zeroed") has no gradient; this smooth version
    does.
    """
    a = cmath.abs(cmath.asarray(amplitude))
    # 20*log10(x) = 20/ln(10) * ln(x); guard against log(0)
    level_db = (20.0 / 2.302585092994046) * cmath.log(a + 1e-12) - 20.0 * _log10(ref)
    return cmath.mean(cmath.relu(cmath.asarray(floor_db) - level_db))


def soft_rank_penalty(query: Any, positive: Any, negatives: Any, *, margin: float = 0.1) -> Any:
    """Soft top-k rank relaxation via cosine-distance margin (Multimodal, 5.1).

    Instead of a hard "positive must rank in top-k", penalise any negative whose
    cosine similarity to the query comes within ``margin`` of the positive's --
    a smooth contrastive relaxation of the rank constraint.
    ``negatives`` is a 2-D array (n_neg x dim).
    """
    q = cmath.flatten(cmath.asarray(query))
    pos = cmath.flatten(cmath.asarray(positive))
    pos_sim = _cosine(q, pos)
    negs = cmath.asarray(negatives)
    n = cmath.shape(negs)[0]
    total = 0.0
    for i in range(n):
        neg_i = cmath.flatten(negs[i])
        total += cmath.to_float(cmath.relu(cmath.asarray(_cosine(q, neg_i) - pos_sim + margin)))
    return cmath.asarray(total / max(1, n))


def _cosine(a: Any, b: Any) -> float:
    denom = cmath.to_float(cmath.norm(a)) * cmath.to_float(cmath.norm(b))
    if denom <= 1e-12:
        return 0.0
    return cmath.to_float(cmath.dot(a, b)) / denom


def _log10(x: float) -> float:
    import math as _m

    return _m.log10(x) if x > 0 else 0.0


# Map from modality -> its default proxy relaxation, for the config's
# ``modality`` preset selection (Section 7.1-7.4).
MODALITY_RELAXATIONS = {
    "vision": soft_iou_penalty,
    "tabular": soft_f1_penalty,
    "audio": spectral_gate_penalty,
    "multimodal": soft_rank_penalty,
}


def register_builtin_relaxations() -> None:
    """Register the modality relaxations as custom assertion types."""
    from .assertions import register_assertion_type

    register_assertion_type("soft_iou", soft_iou_penalty)
    register_assertion_type("soft_f1", soft_f1_penalty)
    register_assertion_type("spectral_gate", spectral_gate_penalty)
    register_assertion_type("soft_rank", soft_rank_penalty)
