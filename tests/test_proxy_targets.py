import numpy as np
import pytest

import compello
from compello.proxy import ModelProxy, TensorProxy
from compello.targets import LogitTarget, OutputTarget


class Dummy:
    def __init__(self):
        self.calls = 0
        self.attr = 42

    def __call__(self, x):
        self.calls += 1
        return x * 2.0

    def custom_method(self):
        return "ok"


def test_wrap_model_returns_model_proxy():
    m = compello.wrap(Dummy())
    assert isinstance(m, ModelProxy)


def test_model_proxy_transparent_passthrough():
    d = Dummy()
    m = compello.wrap(d)
    assert m.attr == 42
    assert m.custom_method() == "ok"
    # calling forwards through and updates the underlying call counter
    m(np.array([1.0]))
    assert d.calls == 1


def test_model_output_is_output_target_and_live():
    m = compello.wrap(Dummy())
    m(np.array([1.0, 2.0]))
    t = m.output
    assert isinstance(t, OutputTarget)
    assert np.allclose(t.tensor, [2.0, 4.0])
    # a second forward refreshes the snapshot
    m(np.array([3.0]))
    assert np.allclose(m.output.tensor, [6.0])


def test_output_before_forward_defers_and_raises_on_eval():
    m = compello.wrap(Dummy())
    # accessing .output before a forward is allowed (deferred, live target)
    t = m.output
    assert isinstance(t, OutputTarget)
    # but evaluating its tensor before any forward raises
    with pytest.raises(RuntimeError):
        _ = t.tensor


def test_wrap_tensor_returns_tensor_proxy_and_indexing_gives_logit_target():
    logits = compello.wrap(np.array([2.0, 1.0, 0.1]))
    assert isinstance(logits, TensorProxy)
    lt = logits[1]
    assert isinstance(lt, LogitTarget)
    assert lt.vocab_index == 1


def test_unwrap_roundtrip():
    d = Dummy()
    m = compello.wrap(d)
    assert compello.unwrap(m) is d
    arr = np.array([1.0])
    tp = compello.wrap(arr)
    assert compello.unwrap(tp) is arr


def test_wrap_is_idempotent():
    m = compello.wrap(Dummy())
    assert compello.wrap(m) is m
