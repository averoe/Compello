"""Declarative YAML/dict configuration (Section 3.4).

A full run's constraint behaviour is expressible as a single declarative config.
This module parses it into a ``CompelloConfig``: a list of ``ConstraintSpec``
(declarative descriptions, bound to live tensors later by a backend adapter) and
a ready-to-use ``ControllerConfig``.

PyYAML is an optional dependency (the ``config`` extra). ``load_config`` accepts
either a path to a YAML file or an already-parsed dict, so config handling never
forces a dependency on code paths that pass dicts directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .controller import ControllerConfig
from .controller.pid import PIDGains

VALID_BACKENDS = {
    "raw_pytorch", "huggingface_trainer", "lightning",
    "tf_gradient_tape", "keras3", "jax_native",
}
VALID_STRATEGIES = {"fixed", "linear_ramp", "adaptive_pid", "dual_ascent"}
VALID_MODALITIES = {"vision", "text", "audio", "tabular", "multimodal"}


@dataclass
class ConstraintSpec:
    name: str
    type: str
    target: str
    condition: Optional[str] = None
    initial_weight: float = 1.0
    respect_loss_mask: bool = False
    transform: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompelloConfig:
    constraints: List[ConstraintSpec]
    controller: ControllerConfig
    backend: str = "raw_pytorch"
    distributed: str = "auto"
    modality: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_preflight_dict(self) -> Dict[str, Any]:
        """Shape expected by ``validation.preflight`` for backend checks."""
        return {
            "backend": self.backend,
            "distributed": self.distributed,
            "diagnostics": self.diagnostics,
            "compiled_optimizer_step": self.diagnostics.get("compiled_optimizer_step", False),
        }


def load_config(source: Any) -> CompelloConfig:
    """Load config from a YAML file path, a YAML string, or a dict."""
    if isinstance(source, dict):
        data = source
    else:
        data = _load_yaml(source)
    return _from_dict(data)


def _load_yaml(source: Any) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to parse YAML config; install the 'config' extra "
            "or pass an already-parsed dict."
        ) from exc
    # path vs raw yaml text
    if isinstance(source, str) and ("\n" in source or ":" in source and not source.endswith((".yml", ".yaml"))):
        return yaml.safe_load(source)
    with open(source, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _from_dict(data: Dict[str, Any]) -> CompelloConfig:
    raw_constraints = data.get("constraints", []) or []
    specs: List[ConstraintSpec] = []
    for rc in raw_constraints:
        known = {"name", "type", "target", "condition", "initial_weight",
                 "respect_loss_mask", "transform"}
        extra = {k: v for k, v in rc.items() if k not in known}
        specs.append(ConstraintSpec(
            name=rc["name"],
            type=rc["type"],
            target=rc.get("target", ""),
            condition=rc.get("condition"),
            initial_weight=float(rc.get("initial_weight", 1.0)),
            respect_loss_mask=bool(rc.get("respect_loss_mask", False)),
            transform=rc.get("transform"),
            extra=extra,
        ))

    cc = data.get("controller", {}) or {}
    strategy = cc.get("strategy", "adaptive_pid")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"unknown controller strategy {strategy!r}")

    gains = PIDGains(
        kp=float(cc.get("kp", 1.0)),
        ki=float(cc.get("ki", 0.01)),
        kd=float(cc.get("kd", 0.1)),
    )
    controller = ControllerConfig(
        strategy=strategy,
        tolerance=float(cc.get("tolerance", 1e-3)),
        patience=int(cc.get("patience", 500)),
        max_steps=int(cc.get("max_steps", 50000)),
        ema_decay=float(cc.get("ema_decay", 0.97)),
        ema_fast_decay=float(cc.get("ema_fast_decay", 0.7)),
        ema_override_steps=int(cc.get("ema_override_steps", 5)),
        ema_baseline_window=int(cc.get("ema_baseline_window", 200)),
        weight_ceiling=float(cc.get("weight_ceiling", 25.0)),
        sharpness_hysteresis=float(cc.get("sharpness_hysteresis", 1.5)),
        gains=gains,
        on_plateau=cc.get("on_plateau", "report_infeasible"),
        max_attempts=int(cc.get("max_attempts", 3)),
        aggressive_momentum_correction=bool(cc.get("aggressive_momentum_correction", False)),
        accumulation_steps=int(cc.get("accumulation_steps", 1)),
        log_space_stability=bool(cc.get("log_space_stability", False)),
        sharpness_patience=int(cc.get("sharpness_patience", 0)),
    )

    backend = data.get("backend", "raw_pytorch")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; valid: {sorted(VALID_BACKENDS)}")

    modality = data.get("modality")
    if modality is not None and modality not in VALID_MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")

    return CompelloConfig(
        constraints=specs,
        controller=controller,
        backend=backend,
        distributed=data.get("distributed", "auto"),
        modality=modality,
        diagnostics=data.get("diagnostics", {}) or {},
    )
