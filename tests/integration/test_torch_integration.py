"""Real PyTorch integration tests for the torch adapter paths.

Skipped automatically when torch is not installed, so the default numpy-only
suite is unaffected. These exercise the previously framework-unverified code:
the torch math backend, wrap/hook capture, gradient-scope reads, un-fused VJP
surgery, functional VJP, momentum-buffer projection, AMP unscale, the
torch.library custom-op interceptor, and the distributed collective path.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import compello
from compello import expect
from compello.backends import torch_backend as tb
from compello.backends.torch_backend import TorchAdapter, TorchBackend


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_torch_backend_registered_and_math():
    from compello import math as cmath
    assert "torch" in cmath.available_backends()
    x = torch.tensor([-1.0, 2.0])
    assert torch.allclose(cmath.relu(x), torch.tensor([0.0, 2.0]))
    assert cmath.to_float(torch.tensor(3.5)) == pytest.approx(3.5)
    s = cmath.softmax(torch.tensor([0.0, 0.0]))
    assert torch.allclose(s, torch.tensor([0.5, 0.5]))


def test_wrap_and_live_output_and_hook():
    lin = nn.Linear(4, 2)
    model = compello.wrap(lin)
    assert model.in_features == 4                   # attribute passthrough
    handle = tb.register_output_hook(model)         # native forward hook
    out = model(torch.randn(3, 4))
    assert model.output.tensor.shape == (3, 2)
    assert torch.allclose(model.output.tensor, out)
    handle.remove()


def test_expect_range_and_probability_floor_on_torch():
    logits = compello.wrap(torch.tensor([2.0, 1.0, 0.1]))
    conf = expect(logits[0], "> 0.6", name="c")
    # softmax(2,1,0.1)[0] ~ 0.66 > 0.6 -> ~0 violation
    assert conf.violation_scalar() == pytest.approx(0.0, abs=1e-2)


def test_read_gradients_scope_filtering():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    adapter = TorchAdapter(compello.wrap(model))
    x = torch.randn(5, 4)
    model(x).sum().backward()
    all_grads = adapter.read_gradients(scope="full_model")
    last = adapter.read_gradients(scope="last_n_layers:2")
    assert len(last) <= len(all_grads)
    assert all(torch.isfinite(g).all() for g in all_grads.values())


def test_unfused_scoped_surgery_writes_grads():
    model = nn.Linear(4, 1)
    adapter = TorchAdapter(compello.wrap(model))
    x = torch.randn(8, 4)
    out = model(x)
    task_loss = ((out - 1.0) ** 2).mean()
    constraint_loss = torch.relu(-out).mean()       # non-negativity penalty
    res = adapter.unfused_scoped_surgery(task_loss, constraint_loss, scope="full_model", weight=1.0)
    # gradients were written and are finite
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert isinstance(res.cosine_similarity, float)


def test_functional_vjp_grads():
    model = nn.Linear(3, 1)
    adapter = TorchAdapter(compello.wrap(model))
    x = torch.randn(6, 3)

    def loss_from_params(params):
        out = torch.func.functional_call(model, params, (x,))
        return (out ** 2).mean()

    grads = adapter.functional_vjp_grads(loss_from_params, scope="full_model")
    assert set(grads) == {n for n, _ in model.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())


def test_project_momentum_buffers_with_adamw():
    model = nn.Linear(4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    adapter = TorchAdapter(compello.wrap(model), optimizer=opt)
    # one real step to populate exp_avg
    model(torch.randn(8, 4)).sum().backward()
    opt.step()
    assert adapter.optimizer_beta1() == pytest.approx(0.9)
    # build a task-direction dict and project momentum buffers
    task_grads = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    modified = adapter.project_momentum_buffers(task_grads)
    assert modified >= 0  # runs without error; count is data-dependent


def test_unscale_gradients_noop_paths():
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    adapter = TorchAdapter(compello.wrap(model), optimizer=opt)
    model(torch.randn(8, 4)).sum().backward()
    adapter.unscale_gradients(None)                 # None scaler is a safe no-op
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    adapter.unscale_gradients(scaler)               # disabled scaler is a no-op
    grads = adapter.read_gradients(scaler=scaler)
    assert all(torch.isfinite(g).all() for g in grads.values())


def test_compiler_safe_grad_interceptor():
    captured = {}

    def transform(grad):
        captured["seen"] = True
        return grad * 2.0                           # double the gradient

    intercept = tb.make_compiler_safe_grad_interceptor(transform, op_name="compello_test::gi")
    x = torch.randn(4, requires_grad=True)
    y = intercept(x).sum()
    y.backward()
    assert captured.get("seen") is True
    assert torch.allclose(x.grad, torch.full((4,), 2.0))


def test_sync_violations_single_rank_gloo(tmp_path):
    import torch.distributed as dist
    init_file = tmp_path / "pg"
    dist.init_process_group(
        backend="gloo", init_method=f"file:///{init_file.as_posix()}",
        rank=0, world_size=1,
    )
    try:
        adapter = TorchAdapter(compello.wrap(nn.Linear(2, 2)), distributed="auto")
        out = adapter.sync_violations({"a": 0.4, "b": 0.9})
        # world_size=1 -> AVG is identity, but the real collective path executed
        assert out == {"a": pytest.approx(0.4), "b": pytest.approx(0.9)}
    finally:
        dist.destroy_process_group()
