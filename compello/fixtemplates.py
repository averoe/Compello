"""Fix templates for the pre-flight shield (Section 8).

Each named error / lint rule maps to an explanation of *why it will hurt the
run* and an *actionable fix* with a before/after code template where one makes
sense. This is what turns a bare error into the "experienced systems engineer"
guidance the pre-flight shield prints.

Keyed by both the trainlint rule name (e.g. "compiled-optimizer-step") and the
named exception (e.g. "CompiledOptimizerStepConflictError") so either surface
can look a template up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FixTemplate:
    title: str
    why: str
    fix: str


_TEMPLATES = {
    # ---- 4.6 ----
    "compiled-optimizer-step": FixTemplate(
        title="CompiledOptimizerStepConflictError",
        why=(
            "PyTorch's AOTAutograd bakes the backward pass into a static compiled graph "
            "before execution. Gradient Surgery requires an eager-mode gap between "
            "backward and the optimizer step to project out conflicting vectors. If they "
            "are compiled together, the tracer strips the surgery logic out and your "
            "constraints are silently ignored."
        ),
        fix=(
            "Split the compiled region: compile forward+backward, keep optimizer.step() eager.\n\n"
            "Change this:\n"
            "  @torch.compile\n"
            "  def full_step(batch):\n"
            "      loss = model(batch).loss\n"
            "      loss.backward()\n"
            "      optimizer.step()\n\n"
            "To this:\n"
            "  @torch.compile\n"
            "  def forward_backward(batch):\n"
            "      return model(batch).loss\n"
            "  # optimizer.step() stays eager so Compello can intercept gradients"
        ),
    ),
    "CompiledOptimizerStepConflictError": None,  # alias, filled below

    # ---- 3.1a ----
    "ambiguous-assertion": FixTemplate(
        title="AmbiguousAssertionError",
        why=(
            "expect() dispatches on the *type* of its target. A raw tensor with no "
            "compello.wrap(...) and no assertion_type= gives the dispatcher nothing to key "
            "on, so it cannot know whether you meant a range, a probability floor, etc."
        ),
        fix=(
            "Wrap the target or state the type explicitly.\n\n"
            "Change this:\n"
            "  expect(raw_tensor, lambda y: y > 0)\n\n"
            "To this:\n"
            "  expect(compello.wrap(raw_tensor), '> 0')\n"
            "  # or: expect(raw_tensor, '> 0', assertion_type='range')"
        ),
    ),
    "AmbiguousAssertionError": None,

    # ---- 4.1b ----
    "distributed-auto-misconfiguration": FixTemplate(
        title="DistributedConfigError",
        why=(
            "JAX has no runtime equivalent of torch.distributed.is_initialized(): its "
            "parallelism is fixed at trace time by the transform, not queryable at runtime. "
            "So 'distributed: auto' cannot detect a JAX distributed run and would silently "
            "skip the cross-device violation sync."
        ),
        fix=(
            "Provide an explicit axis_name for the JAX backend.\n\n"
            "Change this (config.yaml):\n"
            "  backend: jax_native\n"
            "  distributed: auto\n\n"
            "To this:\n"
            "  backend: jax_native\n"
            "  distributed: { axis_name: batch }"
        ),
    ),
    "DistributedConfigError": None,

    # ---- 7.5 ----
    "UnsupportedTargetError": FixTemplate(
        title="UnsupportedTargetError",
        why=(
            "Compello steers a continuous, per-step gradient signal. Tree ensembles have "
            "no differentiable parameters, and scikit-learn's fit() exposes no per-step "
            "gradient hook, so there is nothing for the controller to intervene on."
        ),
        fix=(
            "Use a differentiable substitute, or distill from the tree model.\n\n"
            "  student = NODE(...)  # or FT-Transformer\n"
            "  bridge = compello.distillation_bridge(\n"
            "      teacher=xgboost_model, student=student, constraints=my_constraints)"
        ),
    ),

    # ---- 8.3 PyTorch loop bugs ----
    "zero-grad": FixTemplate(
        title="Missing optimizer.zero_grad()",
        why=(
            "Without zero_grad() before backward(), gradients from previous steps keep "
            "accumulating into .grad, so every step trains on a growing sum of stale "
            "gradients -- a silent correctness bug that looks like instability."
        ),
        fix="Call optimizer.zero_grad() at the top of each training step, before loss.backward().",
    ),
    "detached-loss": FixTemplate(
        title="Undetached loss retained for logging",
        why=(
            "Appending a live loss tensor to a Python list keeps its entire autograd graph "
            "alive; over thousands of steps this leaks memory until the run OOMs hours in."
        ),
        fix="Log the scalar value, not the tensor: losses.append(loss.item())  # or loss.detach()",
    ),
    "untracked-variable": FixTemplate(
        title="Controller state baked as a constant by AutoGraph",
        why=(
            "A plain Python attribute mutated inside @tf.function is traced once and frozen "
            "as a constant, so the controller appears to update on the first trace and then "
            "never changes again."
        ),
        fix=(
            "Store controller state in a tf.Variable and mutate via assign.\n\n"
            "Change this:\n"
            "  self.error_integral += x\n\n"
            "To this:\n"
            "  self.error_integral = tf.Variable(0.0, trainable=False)\n"
            "  self.error_integral.assign_add(x)"
        ),
    ),
    "impure-mutation": FixTemplate(
        title="In-place mutation inside a jitted function",
        why=(
            "jax.jit requires pure functions. Mutating an object in place either raises "
            "ConcretizationTypeError or silently freezes the controller's view of state."
        ),
        fix=(
            "Thread state through an explicit PyTree return value instead of mutating.\n\n"
            "  grads, controller_state = cjax.steer_step(\n"
            "      grads, preds, controller_state, constraints=...)"
        ),
    ),
}

# resolve aliases (exception-name -> same template as the rule)
_TEMPLATES["CompiledOptimizerStepConflictError"] = _TEMPLATES["compiled-optimizer-step"]
_TEMPLATES["AmbiguousAssertionError"] = _TEMPLATES["ambiguous-assertion"]
_TEMPLATES["DistributedConfigError"] = _TEMPLATES["distributed-auto-misconfiguration"]


def get_fix_template(key: str) -> Optional[FixTemplate]:
    return _TEMPLATES.get(key)
