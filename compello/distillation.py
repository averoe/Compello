"""Distillation bridge for classical-ML surrogates (Section 7.5).

Workflow: train an XGBoost/LightGBM teacher normally for best raw accuracy, then
train a differentiable student (NODE / FT-Transformer / MLP) with a combined
objective -- a distillation loss matching the teacher's output distribution,
*plus* whichever Compello constraints matter, applied to the student via the
normal ``expect()`` DSL. No new controller mechanics are required: the
distillation loss is just another term the constraint-weighting logic in Section
4 already balances against.

``distillation_bridge`` is the framework-independent orchestrator. The actual
teacher inference and student backward pass are supplied by the caller as small
callables so this stays backend-agnostic and testable without any DL framework:
the caller provides how to get teacher predictions and how to compute the
student's distillation loss; Compello combines it with the constraint penalties
and returns the total objective plus per-term telemetry each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import math as cmath
from .assertions import Assertion
from .controller.controller import Controller, ControllerConfig


@dataclass
class DistillationStep:
    total: float
    distillation: float
    constraint_terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class DistillationBridge:
    """Combines a distillation loss with weighted constraint penalties (7.5)."""

    teacher: Any
    student: Any
    constraints: Sequence[Assertion]
    controller: Controller = field(default=None)
    distill_weight: float = 1.0

    def __post_init__(self):
        if self.controller is None:
            self.controller = Controller(ControllerConfig())
        self.controller.register_assertions(self.constraints)

    def combined_loss(
        self,
        distillation_loss: Any,
        *,
        constraint_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DistillationStep:
        """Return total objective = distill_weight * distill + sum(w_i * penalty_i).

        ``distillation_loss`` is a backend-native scalar (e.g. KL/MSE between
        student and teacher outputs) computed by the caller. ``constraint_inputs``
        optionally supplies per-constraint evaluation kwargs.
        """
        constraint_inputs = constraint_inputs or {}
        distill_f = cmath.to_float(distillation_loss)
        total = self.distill_weight * distill_f
        terms: Dict[str, float] = {}
        raw_violations: Dict[str, float] = {}
        for a in self.constraints:
            inputs = constraint_inputs.get(a.name, {})
            v = a.violation_scalar(**inputs)
            raw_violations[a.name] = v
            w = self.controller.states[a.name].weight
            terms[a.name] = w * v
            total += w * v
        # advance the controller so weights adapt exactly as in a normal run
        self.controller.step(raw_violations)
        return DistillationStep(total=total, distillation=distill_f, constraint_terms=terms)


def distillation_bridge(
    *,
    teacher: Any,
    student: Any,
    constraints: Sequence[Assertion],
    controller: Optional[Controller] = None,
    distill_weight: float = 1.0,
) -> DistillationBridge:
    """Entry point matching the plan's
    ``compello.distillation_bridge(teacher=..., student=..., constraints=...)``.
    """
    return DistillationBridge(
        teacher=teacher, student=student, constraints=list(constraints),
        controller=controller, distill_weight=distill_weight,
    )
