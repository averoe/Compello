"""PyTorch backend adapter -- the load-bearing reference (Sections 4, 10.5).

Requires ``torch``. Provides:
  * ``TorchBackend`` -- the ``compello.math`` array-op implementation for
    ``torch.Tensor``, auto-registered on import.
  * ``TorchAdapter`` -- batched DDP/FSDP violation sync (4.1), post-backward
    gradient access for surgery (4.6), optimizer beta1 readout (4.7).
  * ``CompelloTrainerCallback`` -- HuggingFace ``TrainerCallback`` integration.
  * ``register_output_hook`` -- the native ``register_forward_hook`` capture
    referenced by Section 3.0.

NOTE: This module cannot run without PyTorch installed. It is structured so that
it drops straight onto the framework-independent controller/assertions/math core
that IS tested here against the numpy backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import BackendNotAvailableError

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    torch = None
    _HAS_TORCH = False


def _require():
    if not _HAS_TORCH:
        raise BackendNotAvailableError(
            "compello.backends.torch requires PyTorch. Install with "
            "`pip install compello[torch]`."
        )


class TorchBackend:
    name = "torch"

    @staticmethod
    def is_available() -> bool:
        return _HAS_TORCH

    def asarray(self, x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

    def to_float(self, x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().mean()) if x.numel() > 1 else float(x.detach())
        return float(x)

    def is_native(self, x) -> bool:
        return _HAS_TORCH and isinstance(x, torch.Tensor)

    def shape(self, x):
        return tuple(x.shape)

    def maximum(self, a, b):
        return torch.maximum(self.asarray(a), self.asarray(b))

    def minimum(self, a, b):
        return torch.minimum(self.asarray(a), self.asarray(b))

    def abs(self, x):
        return torch.abs(self.asarray(x))

    def exp(self, x):
        return torch.exp(self.asarray(x))

    def log(self, x):
        return torch.log(self.asarray(x))

    def clip(self, x, lo, hi):
        return torch.clamp(self.asarray(x), lo, hi)

    def sigmoid(self, x):
        return torch.sigmoid(self.asarray(x))

    def relu(self, x):
        return torch.relu(self.asarray(x))

    def sum(self, x, axis=None):
        return torch.sum(self.asarray(x), dim=axis) if axis is not None else torch.sum(self.asarray(x))

    def mean(self, x, axis=None):
        return torch.mean(self.asarray(x), dim=axis) if axis is not None else torch.mean(self.asarray(x))

    def dot(self, a, b):
        return torch.dot(self.asarray(a).reshape(-1), self.asarray(b).reshape(-1))

    def norm(self, x):
        return torch.linalg.norm(self.asarray(x).reshape(-1))

    def stack(self, xs, axis=0):
        return torch.stack([self.asarray(x) for x in xs], dim=axis)

    def flatten(self, x):
        return self.asarray(x).reshape(-1)

    def softmax(self, x, axis=-1):
        return torch.softmax(self.asarray(x), dim=axis)

    def diff(self, x, axis=-1):
        return torch.diff(self.asarray(x), dim=axis)


class TorchAdapter:
    """Training-loop adapter implementing the Section 4 safety mechanisms."""

    name = "torch"

    def __init__(self, model, optimizer=None, *, distributed: str = "auto"):
        _require()
        self.model = model
        self.optimizer = optimizer
        self.distributed = distributed

    # -- 4.1: batched DDP/FSDP violation sync --------------------------
    def sync_violations(self, local_violations: Dict[str, float]) -> Dict[str, float]:
        if not self._is_distributed():
            return dict(local_violations)
        from .sync import batched_sync

        def reduce_fn(values):
            # one collective per step, regardless of constraint count (the v2 fix)
            stacked = torch.tensor(values, dtype=torch.float32)
            if torch.cuda.is_available():
                stacked = stacked.cuda()
            torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.AVG)
            return [float(x) for x in stacked]

        return batched_sync(local_violations, reduce_fn)

    def _is_distributed(self) -> bool:
        if self.distributed == "off":
            return False
        try:
            return torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            return False

    # -- 1.3: AMP GradScaler-safe unscaling ----------------------------
    def unscale_gradients(self, scaler) -> None:
        """Unscale the optimizer's gradients in place before Compello reads them.

        Under AMP the backward pass produces gradients multiplied by the
        scaler's loss-scale factor. Gradient Surgery and the violation signal
        must operate on *raw, unscaled* gradients, or the projection geometry
        and the multiplier update would both be distorted by the (arbitrary,
        dynamically-changing) scale factor. Calling ``scaler.unscale_(optimizer)``
        exactly once, here, restores the true gradients; it is idempotent per
        step and leaves the scaler's overflow bookkeeping intact so the normal
        ``scaler.step()/update()`` path still skips NaN/inf steps correctly.
        """
        if scaler is not None and self.optimizer is not None:
            scaler.unscale_(self.optimizer)

    # -- 4.6: post-backward, pre-step gradient access ------------------
    def read_gradients(self, scope: Optional[str] = None, *, scaler=None) -> Dict[str, Any]:
        if scaler is not None:
            self.unscale_gradients(scaler)
        params = list(self.model.named_parameters())
        selected = _apply_scope(params, scope)
        # skip params whose grads are non-finite (an AMP overflow step): the
        # scaler will skip the optimizer step anyway, so surgery must not act.
        return {
            name: p.grad for name, p in selected
            if p.grad is not None and bool(torch.isfinite(p.grad).all())
        }

    def write_gradients(self, grads: Dict[str, Any]) -> None:
        table = dict(self.model.named_parameters())
        for name, g in grads.items():
            if name in table and table[name].grad is not None:
                table[name].grad.copy_(g)

    # -- 1.1: un-fused, layer-scoped gradient surgery ------------------
    def unfused_scoped_surgery(self, task_loss, constraint_loss, *,
                               scope: Optional[str] = None, weight: float = 1.0):
        """Isolate the task and constraint gradients on *independent* backward
        paths, project the conflict out per Section 5.2, and write the combined
        gradient -- restricted to the in-scope parameters (5.6).

        The Gradient Fusion Paradox (1.1): if you form ``L_task + w*L_constraint``
        and call one ``.backward()``, the two directional forces are already
        summed by the time they reach ``.grad`` and cannot be separated. Here the
        two losses are differentiated separately (``torch.autograd.grad`` gives
        two independent VJPs against the shared graph, with ``retain_graph`` for
        the first), so the per-parameter task and constraint gradients exist
        distinctly before they are recombined -- which is exactly what PCGrad
        needs. Restricting to the in-scope parameter subset bounds the extra
        backward cost to that fraction rather than a full 2x global pass.
        """
        from ..diagnostics.surgery import scoped_gradient_surgery, select_in_scope

        named = list(self.model.named_parameters())
        in_scope = select_in_scope([n for n, _ in named], scope)
        names = [n for n, _ in named if n in in_scope]
        params = [p for n, p in named if n in in_scope]

        g_task = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
        g_con = torch.autograd.grad(constraint_loss, params, retain_graph=False, allow_unused=True)
        task_grads = {n: (g if g is not None else torch.zeros_like(p))
                      for n, g, p in zip(names, g_task, params)}
        con_grads = {n: (g if g is not None else torch.zeros_like(p))
                     for n, g, p in zip(names, g_con, params)}

        res = scoped_gradient_surgery(task_grads, con_grads, scope=None)
        table = dict(named)
        for n in names:
            combined = task_grads[n] + weight * res.corrected[n]
            if table[n].grad is None:
                table[n].grad = combined.detach().clone()
            else:
                table[n].grad.copy_(combined)
        return res

    def functional_vjp_grads(self, loss_from_params, *, scope: Optional[str] = None) -> Dict[str, Any]:
        """Pure functional-VJP isolation of a single loss's gradient (1.1).

        ``loss_from_params(params_dict) -> scalar`` is differentiated via
        ``torch.func.vjp`` against only the in-scope parameters, so its gradient
        path is tracked independently of any other loss term and never fuses in
        the shared autograd graph. Returns ``{param_name: grad}``.
        """
        from torch.func import vjp

        named = dict(self.model.named_parameters())
        from ..diagnostics.surgery import select_in_scope

        names = list(select_in_scope(list(named), scope))
        primal = {n: named[n].detach() for n in names}
        out, vjp_fn = vjp(loss_from_params, primal)
        cotangent = torch.ones_like(out)
        (grads,) = vjp_fn(cotangent)
        return grads

    # -- 4.7: optimizer first-moment decay (read once, not live) -------
    def optimizer_beta1(self) -> Optional[float]:
        if self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            betas = group.get("betas")
            if betas:
                return float(betas[0])
            if "momentum" in group:  # SGD-with-momentum style
                return float(group["momentum"])
        return None

    # -- 4.7b: aggressive direct momentum-buffer surgery (opt-in) ------
    def project_momentum_buffers(self, task_grads: Dict[str, Any]) -> int:
        """Project the conflicting component out of the optimizer's first-moment
        buffers (``exp_avg``) as well (Section 4.7, ``aggressive_momentum_correction``).

        This is the version-fragile path Compello ships OFF by default: it reaches
        into ``optimizer.state[p]['exp_avg']``, whose key name/shape differs across
        optimizers and PyTorch versions and does not exist for SGD-without-moments.
        For each in-scope parameter whose momentum buffer conflicts with the task
        direction (negative inner product), it removes the task-direction component
        from the buffer in place. Returns the number of buffers modified.
        """
        if self.optimizer is None:
            return 0
        from ..diagnostics.surgery import project_out_conflict

        name_by_param = {p: n for n, p in self.model.named_parameters()}
        modified = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p, {})
                exp_avg = state.get("exp_avg")
                name = name_by_param.get(p)
                if exp_avg is None or name is None or name not in task_grads:
                    continue
                corrected, changed = project_out_conflict(
                    exp_avg, task_grads[name], only_if_conflicting=True)
                if changed:
                    exp_avg.copy_(corrected)
                    modified += 1
        return modified


class CompelloLightningCallback:
    """PyTorch Lightning integration (Section 3.3).

    Implemented against Lightning's existing callback hooks
    (``on_train_batch_end``, ``on_validation_epoch_end``) without touching
    Lightning internals. Not a hard subclass of ``lightning.pytorch.Callback``
    so the module imports without Lightning installed; register it on the
    ``Trainer(callbacks=[...])`` list, where duck typing on the hook names is all
    Lightning requires.
    """

    def __init__(self, controller, assertions, adapter: "TorchAdapter"):
        _require()
        self.controller = controller
        self.assertions = list(assertions)
        self.adapter = adapter
        self.controller.register_assertions(self.assertions)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # pragma: no cover
        local = {a.name: a.violation_scalar() for a in self.assertions}
        synced = self.adapter.sync_violations(local)
        self.controller.step(synced)


def make_compiler_safe_grad_interceptor(transform, *, op_name: str = "compello::grad_intercept"):
    """Build a ``torch.compile``-safe activation-gradient interceptor (Section 4.6).

    For the narrow case where an assertion must inspect or modify an *intermediate
    activation's* gradient, a raw ``tensor.register_hook`` either forces a graph
    break or is silently dropped by the compiled graph tracer. Instead we define
    an identity ``custom_op`` whose *backward* applies ``transform(grad)``.
    Because it is registered through ``torch.library``, ``AOTAutograd`` traces it
    as a first-class op and it survives compilation with no graph break.

    Insert the returned callable at the activation you want to steer::

        h = model.layer               # some intermediate
        h = grad_intercept(h)          # gradient flowing back through h is transformed

    ``transform(grad_tensor) -> grad_tensor`` is your projection/scaling. Requires
    a PyTorch new enough to expose ``torch.library.custom_op`` (2.4+).
    """
    _require()
    from torch.library import custom_op, register_autograd

    # Real type annotations are required: torch.library.infer_schema reads the
    # function's annotations, so they must be the actual torch.Tensor type, not
    # a string forward-reference.
    def _grad_intercept_impl(x):
        return x.clone()

    _grad_intercept_impl.__annotations__ = {"x": torch.Tensor, "return": torch.Tensor}
    grad_intercept = custom_op(op_name, mutates_args=())(_grad_intercept_impl)

    def _grad_intercept_fake(x):
        return torch.empty_like(x)

    grad_intercept.register_fake(_grad_intercept_fake)

    def _backward(ctx, grad):
        return transform(grad)

    def _setup_context(ctx, inputs, output):
        return None

    register_autograd(op_name, _backward, setup_context=_setup_context)
    return grad_intercept


def register_output_hook(wrapped_model) -> Any:
    """Attach a native forward hook that refreshes the proxy's output snapshot.

    Referenced by Section 3.0: the hook fires around the (possibly compiled)
    forward callable and observes its return value, so it introduces no graph
    break. ``wrapped_model`` is a ``compello.proxy.ModelProxy``.
    """
    _require()
    underlying = wrapped_model.unwrap()

    def _hook(_module, _inputs, output):
        object.__setattr__(wrapped_model, "_compello_last_output", output)

    return underlying.register_forward_hook(_hook)


class CompelloTrainerCallback:
    """HuggingFace ``TrainerCallback`` integration (Section 3.3).

    Reads loss/logits and injects the penalty term through the callback hooks
    (``on_step_end`` etc.) without touching Trainer internals. Constructed with
    a ready ``Controller`` and the declared assertions.
    """

    def __init__(self, controller, assertions, adapter: "TorchAdapter"):
        _require()
        self.controller = controller
        self.assertions = list(assertions)
        self.adapter = adapter
        self.controller.register_assertions(self.assertions)

    def on_step_end(self, args, state, control, **kwargs):  # pragma: no cover
        local = {a.name: a.violation_scalar() for a in self.assertions}
        synced = self.adapter.sync_violations(local)
        self.controller.step(synced)
        return control


def register() -> None:
    """Register the torch math backend with ``compello.math`` if available."""
    if _HAS_TORCH:
        from .. import math as cmath

        cmath.register_backend(TorchBackend())


def _apply_scope(named_params: List, scope: Optional[str]):
    """Implement gradient_surgery_scope (5.6) using the shared, backend-agnostic
    scope resolver so PyTorch and the portable diagnostics agree exactly."""
    from ..diagnostics.surgery import select_in_scope

    names = [n for n, _ in named_params]
    in_scope = select_in_scope(names, scope)
    return [(n, p) for n, p in named_params if n in in_scope]


register()
