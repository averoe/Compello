"""Tests for the production-hardening additions (Section 1.x, 5.6, 6.2, 7.5)."""

import json
import math
import os

import numpy as np
import pytest

import compello
from compello import expect
from compello.controller import Controller, ControllerConfig
from compello.controller.sharpness import update_sharpness
from compello.controller.state import ConstraintState
from compello.diagnostics import scoped_gradient_surgery, select_in_scope, full_model_scope_warning


# --- 1.4 gradient-accumulation freeze --------------------------------------

def test_micro_step_freezes_until_macro_boundary():
    cfg = ControllerConfig(strategy="adaptive_pid", accumulation_steps=4, tolerance=0.0)
    c = Controller(cfg)
    c.register("k", 1.0)
    w0 = c.states["k"].weight
    # first 3 micro-batches: no update, no controller step advance
    assert c.micro_step({"k": 1.0}) is None
    assert c.micro_step({"k": 1.0}) is None
    assert c.micro_step({"k": 1.0}) is None
    assert c.states["k"].weight == w0
    assert c.step_index == 0
    # 4th micro-batch triggers exactly one real step at the macro boundary
    result = c.micro_step({"k": 1.0})
    assert result is not None
    assert c.step_index == 1


def test_micro_step_averages_violations():
    cfg = ControllerConfig(accumulation_steps=2, tolerance=0.0)
    c = Controller(cfg)
    c.register("k", 1.0)
    c.micro_step({"k": 0.0})
    result = c.micro_step({"k": 1.0})   # macro boundary; averaged violation = 0.5
    assert result.per_constraint["k"].raw_violation == pytest.approx(0.5)


# --- 1.5 hysteresis dead-band patience decay -------------------------------

def test_deadband_patience_eventually_rearms():
    st = ConstraintState("p")
    st.alpha = 10.0
    kw = dict(g_floor=1e-4, sharpness_hysteresis=1.5, sharpness_patience=5)
    # collapse -> disarm
    update_sharpness(st, 1e-7, metric_satisfied=False, **kw)
    assert st.sharpness_armed is False
    # park the grad norm INSIDE the dead-band (between floor and floor*1.5).
    # Without patience decay this would lock forever; with it, it re-arms.
    band_value = 1e-4 * 1.2
    rearmed = False
    for _ in range(30):
        update_sharpness(st, band_value, metric_satisfied=False, **kw)
        if st.sharpness_armed:
            rearmed = True
            break
    assert rearmed


def test_deadband_no_decay_when_patience_zero():
    st = ConstraintState("p")
    st.alpha = 10.0
    kw = dict(g_floor=1e-4, sharpness_hysteresis=1.5, sharpness_patience=0)
    update_sharpness(st, 1e-7, metric_satisfied=False, **kw)
    for _ in range(50):
        update_sharpness(st, 1e-4 * 1.2, metric_satisfied=False, **kw)
    assert st.sharpness_armed is False  # stays locked (no patience decay)


# --- 1.3 log-space multiplier stability ------------------------------------

def test_log_space_keeps_weight_positive_and_finite():
    cfg = ControllerConfig(strategy="adaptive_pid", log_space_stability=True,
                           tolerance=0.0, weight_ceiling=1e9)
    c = Controller(cfg)
    c.register("k", 1.0)
    for _ in range(200):
        c.step({"k": 1e-9})   # tiny violations that would underflow additively
    w = c.states["k"].weight
    assert w > 0.0 and math.isfinite(w)


# --- 5.6 backend-agnostic layer-scoped surgery -----------------------------

def test_select_in_scope_variants():
    names = ["l0.w", "l1.w", "l2.w", "l3.w"]
    assert select_in_scope(names, "full_model") == set(names)
    assert select_in_scope(names, "last_n_layers:2") == {"l2.w", "l3.w"}
    assert select_in_scope(names, "modules:l1,l3") == {"l1.w", "l3.w"}


def test_scoped_surgery_only_touches_in_scope_and_orthogonalizes():
    task = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    con = {"a": np.array([-1.0, 1.0]), "b": np.array([-1.0, 1.0])}
    res = scoped_gradient_surgery(task, con, scope="modules:b")
    # 'a' untouched, 'b' projected orthogonal to task['b']
    assert np.allclose(res.corrected["a"], con["a"])
    assert float(np.dot(res.corrected["b"], task["b"])) == pytest.approx(0.0, abs=1e-9)
    assert res.projected is True
    assert res.cost_fraction == pytest.approx(0.5)


def test_scoped_surgery_matches_concatenated_projection():
    # joint projection over both params must equal concatenated-vector PCGrad
    task = {"a": np.array([1.0, 2.0]), "b": np.array([0.5])}
    con = {"a": np.array([-2.0, 1.0]), "b": np.array([-1.0])}
    res = scoped_gradient_surgery(task, con, scope="full_model")
    tcat = np.concatenate([task["a"], task["b"]])
    ccat = np.concatenate([con["a"], con["b"]])
    coeff = np.dot(ccat, tcat) / np.dot(tcat, tcat)
    exp_a = con["a"] - coeff * task["a"]
    exp_b = con["b"] - coeff * task["b"]
    assert np.allclose(res.corrected["a"], exp_a)
    assert np.allclose(res.corrected["b"], exp_b)


def test_full_model_scope_warning():
    assert full_model_scope_warning("full_model", 2_000_000_000) is not None
    assert full_model_scope_warning("full_model", 100_000_000) is None
    assert full_model_scope_warning("last_n_layers:8", 2_000_000_000) is None


def test_preflight_emits_full_model_cost_warning():
    report = compello.preflight(
        [], {"diagnostics": {"gradient_surgery_scope": "full_model"},
             "param_count": 3_000_000_000})
    assert any("full_model" in c.kind for c in report.conflicts)


# --- 6.2 checkpoint serialization ------------------------------------------

def test_json_checkpoint_roundtrip(tmp_path):
    cfg = ControllerConfig()
    c = Controller(cfg)
    c.register("a", 1.0)
    c.register("b", 2.0)
    for _ in range(30):
        c.step({"a": 0.5, "b": 0.1})
    path = os.path.join(tmp_path, "ckpt.json")
    compello.save_controller(c, path)
    c2 = compello.load_controller(path, cfg)
    assert c2.to_dict() == c.to_dict()
    assert c2.weights == c.weights


def test_npz_checkpoint_roundtrip(tmp_path):
    cfg = ControllerConfig()
    c = Controller(cfg)
    c.register("x", 1.5)
    for _ in range(10):
        c.step({"x": 0.2})
    path = os.path.join(tmp_path, "ckpt.npz")
    compello.save_controller(c, path)
    c2 = compello.load_controller(path, cfg)
    assert c2.weights == c.weights


# --- 7.5 NODE / FT-Transformer numpy reference forwards --------------------

def test_node_reference_forward_shape_and_finite():
    node = compello.NodeReference(in_features=6, out_features=3, n_trees=4, depth=3, seed=1)
    x = np.random.default_rng(0).normal(size=(5, 6))
    out = node.forward(x)
    assert out.shape == (5, 3)
    assert np.isfinite(out).all()


def test_ft_transformer_reference_forward_shape_and_finite():
    ft = compello.FTTransformerReference(in_features=8, out_features=2, d_token=16, n_heads=2, seed=1)
    x = np.random.default_rng(0).normal(size=(4, 8))
    out = ft.forward(x)
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_build_node_numpy_backend():
    m = compello.build_node(4, 1, backend="numpy", n_trees=2, depth=2)
    assert m.forward(np.zeros((3, 4))).shape == (3, 1)


# --- 6.6 cooper objects path (no cooper installed here) --------------------

def test_export_to_cooper_objects_without_cooper():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.6", name="conf")
    export, multipliers = compello.export_to_cooper_objects([a])
    assert multipliers is None                     # cooper not installed
    assert any("not installed" in n for n in export.notes)
    assert export.constraints[0].name == "conf"
