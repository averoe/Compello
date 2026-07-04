"""Steerability check for assertion targets (Section 7.5).

Compello can only steer a model that exposes a continuous, per-step gradient
interface. Tree ensembles (RandomForest, XGBoost, LightGBM, CatBoost) have no
differentiable parameters, and scikit-learn's own gradient-based estimators
(``SGDClassifier``/``SGDRegressor``) do SGD internally but expose only an opaque
``fit()`` with no per-step hook. Both must fail with a clear, named error rather
than deep inside the controller.

Detection is by module/class name so it needs no framework imported -- the check
runs in ``trainlint``/``preflight`` before any training starts.
"""

from __future__ import annotations

from typing import Any

from .exceptions import UnsupportedTargetError
from .proxy import ModelProxy, unwrap

# (module_prefix, reason) pairs for libraries with no per-step gradient surface
_NON_DIFF_MODULE_PREFIXES = (
    ("xgboost", "XGBoost boosters are built by discrete tree splitting; no gradient interface."),
    ("lightgbm", "LightGBM boosters are built by discrete tree splitting; no gradient interface."),
    ("catboost", "CatBoost models are built by discrete tree splitting; no gradient interface."),
)

# scikit-learn estimators that are non-differentiable OR expose no per-step hook
_SKLEARN_UNSUPPORTED = {
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    # gradient-based internally, but fit() is opaque -- no per-step interface
    "SGDClassifier", "SGDRegressor",
}


def _type_info(obj: Any):
    t = type(obj)
    module = getattr(t, "__module__", "") or ""
    return module, t.__name__


def is_steerable(model: Any) -> bool:
    """Return True if ``model`` plausibly exposes a differentiable interface."""
    try:
        check_steerable(model)
        return True
    except UnsupportedTargetError:
        return False


def check_steerable(model: Any) -> None:
    """Raise ``UnsupportedTargetError`` if ``model`` cannot be steered (7.5)."""
    target = unwrap(model) if isinstance(model, ModelProxy) else model
    module, name = _type_info(target)
    root = module.split(".")[0]

    for prefix, reason in _NON_DIFF_MODULE_PREFIXES:
        if root == prefix:
            raise UnsupportedTargetError(
                f"{name} ({module}) cannot be steered: {reason} Use a "
                f"differentiable substitute (NODE / FT-Transformer) or "
                f"compello.distillation_bridge(...) (Section 7.5)."
            )

    if root == "sklearn" and name in _SKLEARN_UNSUPPORTED:
        if name.startswith("SGD"):
            raise UnsupportedTargetError(
                f"{name} does SGD internally but scikit-learn's fit() exposes no "
                f"per-step gradient interface for Compello to hook (Section 7.5). "
                f"Reimplement as a differentiable linear layer instead."
            )
        raise UnsupportedTargetError(
            f"{name} is a non-differentiable tree model; no gradient signal to "
            f"steer (Section 7.5). Use NODE/FT-Transformer or distillation_bridge."
        )

    # Generic heuristic: an object with fit()/predict() but no differentiable
    # parameter surface (parameters / trainable_variables) and not callable.
    has_fit = callable(getattr(target, "fit", None))
    has_predict = callable(getattr(target, "predict", None))
    looks_callable_model = callable(target) or callable(getattr(target, "__call__", None))
    has_params = (
        hasattr(target, "parameters")
        or hasattr(target, "trainable_variables")
        or hasattr(target, "apply")  # flax-style
    )
    if has_fit and has_predict and not has_params and not looks_callable_model:
        raise UnsupportedTargetError(
            f"{name} ({module}) exposes fit()/predict() but no differentiable "
            f"parameter surface; Compello needs a per-step gradient interface "
            f"(Section 7.5)."
        )
