"""Tests for the backend-agnostic logic extracted out of the framework adapters:
the PCGrad projection kernel (4.7b/5.2) and the batched distributed-sync
discipline (4.1). These are the parts that were previously only exercisable
with a framework installed; extracting them makes the core logic testable with
numpy while the adapters keep only thin framework-API wrappers.
"""

import numpy as np
import pytest

from compello.backends.sync import batched_sync
from compello.diagnostics import project_out_conflict


# --- project_out_conflict (shared PCGrad kernel) ---------------------------

def test_project_removes_direction_component():
    v = np.array([-1.0, 1.0])
    d = np.array([1.0, 0.0])          # v conflicts with d (negative inner)
    corrected, changed = project_out_conflict(v, d, only_if_conflicting=True)
    assert changed is True
    assert float(np.dot(corrected, d)) == pytest.approx(0.0, abs=1e-9)


def test_project_skips_when_not_conflicting():
    v = np.array([1.0, 1.0])
    d = np.array([1.0, 0.0])          # aligned (positive inner) -> untouched
    corrected, changed = project_out_conflict(v, d, only_if_conflicting=True)
    assert changed is False
    assert np.allclose(corrected, v)


def test_project_unconditional_orthogonalizes():
    v = np.array([2.0, 1.0])
    d = np.array([1.0, 0.0])
    corrected, changed = project_out_conflict(v, d, only_if_conflicting=False)
    assert changed is True
    assert float(np.dot(corrected, d)) == pytest.approx(0.0, abs=1e-9)


def test_project_zero_direction_is_noop():
    v = np.array([1.0, 1.0])
    d = np.array([0.0, 0.0])
    corrected, changed = project_out_conflict(v, d)
    assert changed is False
    assert np.allclose(corrected, v)


def test_project_preserves_shape_2d():
    v = np.array([[-1.0, 1.0], [2.0, -2.0]])
    d = np.array([[1.0, 0.0], [0.0, 1.0]])
    corrected, changed = project_out_conflict(v, d, only_if_conflicting=False)
    assert corrected.shape == v.shape
    assert float(np.dot(corrected.ravel(), d.ravel())) == pytest.approx(0.0, abs=1e-9)


# --- batched_sync (one collective per step) --------------------------------

def test_batched_sync_single_reduce_call_and_unpack():
    calls = {"n": 0}

    def reduce_fn(values):
        calls["n"] += 1
        # simulate an AVG all_reduce across 2 replicas by returning the input
        # (already local mean) -- identity keeps the assertion simple
        return [v for v in values]

    local = {"a": 0.2, "b": 0.5, "c": 0.9}
    out = batched_sync(local, reduce_fn)
    assert calls["n"] == 1                     # exactly one collective, 3 constraints
    assert out == {"a": 0.2, "b": 0.5, "c": 0.9}


def test_batched_sync_applies_reduction_and_preserves_order():
    def reduce_fn(values):
        return [v * 2.0 for v in values]       # a stand-in reduction

    out = batched_sync({"x": 1.0, "y": 2.0}, reduce_fn)
    assert out == {"x": 2.0, "y": 4.0}


def test_batched_sync_empty():
    assert batched_sync({}, lambda v: v) == {}


def test_batched_sync_length_mismatch_raises():
    with pytest.raises(ValueError):
        batched_sync({"a": 1.0, "b": 2.0}, lambda v: v[:1])
