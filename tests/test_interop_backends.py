import numpy as np
import pytest

import compello
from compello import expect
from compello.controller import Controller, ControllerConfig


def test_cooper_export_snapshots_multipliers_and_lists_dropped_state():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.6", name="conf")
    c = Controller(ControllerConfig())
    c.register("conf", initial_weight=3.5)
    export = compello.export_to_cooper([a], controller=c)
    assert len(export.constraints) == 1
    ce = export.constraints[0]
    assert ce.multiplier == pytest.approx(3.5)
    # EMA / dual-rate / grace-window state must be reported as dropped (6.6)
    assert any("ema" in d for d in ce.dropped)
    assert any("momentum" in d for d in ce.dropped)
    assert "multiplier" in ce.transferred


def test_cooper_export_summary_renders():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.6", name="conf")
    export = compello.export_to_cooper([a])
    text = export.summary()
    assert "conf" in text and "multiplier" in text


def test_dl_backends_available_is_consistent():
    import compello.backends as b
    # available() reports whichever deep-learning backends are importable; it
    # must be a subset of the known names and agree with the adapters' own flags.
    avail = set(b.available())
    assert avail <= {"torch", "tensorflow", "jax"}
    from compello.backends import torch_backend, tf_backend, jax_backend
    assert ("torch" in avail) == torch_backend._HAS_TORCH
    assert ("tensorflow" in avail) == tf_backend._HAS_TF
    assert ("jax" in avail) == jax_backend._HAS_JAX


def test_torch_adapter_requires_torch():
    from compello.backends.torch_backend import TorchAdapter, _HAS_TORCH
    if not _HAS_TORCH:
        with pytest.raises(compello.BackendNotAvailableError):
            TorchAdapter(model=object())


def test_jax_distributed_check_rejects_auto():
    from compello.backends import jax_backend
    if jax_backend._HAS_JAX:
        with pytest.raises(compello.DistributedConfigError):
            jax_backend.check_distributed_config("auto", None)
    else:
        pytest.skip("jax not installed")
