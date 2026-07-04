"""Tests for the second-pass gap-closure features."""

import numpy as np
import pytest

import compello
from compello import expect
from compello.controller import Controller, ControllerConfig
from compello.exceptions import UnsupportedTargetError


# --- module shims (1) ------------------------------------------------------

def test_compello_tf_and_jax_modules_import():
    import compello.tf as ctf
    import compello.jax as cjax
    assert hasattr(ctf, "ConstraintTape")
    assert hasattr(cjax, "steer_step")
    assert hasattr(cjax, "init_controller_state")


# --- non-differentiable target detection (2) -------------------------------

class _FakeXGB:
    __module__ = "xgboost.sklearn"

    def fit(self, *a):
        ...

    def predict(self, *a):
        ...


class _FakeRF:
    __module__ = "sklearn.ensemble"

    def fit(self, *a):
        ...

    def predict(self, *a):
        ...


_FakeRF.__name__ = "RandomForestClassifier"
_FakeXGB.__name__ = "XGBClassifier"


def test_tree_ensemble_rejected():
    assert compello.is_steerable(_FakeRF()) is False
    with pytest.raises(UnsupportedTargetError):
        compello.check_steerable(_FakeXGB())


def test_sgdclassifier_rejected():
    class _SGD:
        __module__ = "sklearn.linear_model"

        def fit(self, *a): ...
        def predict(self, *a): ...
    _SGD.__name__ = "SGDClassifier"
    with pytest.raises(UnsupportedTargetError):
        compello.check_steerable(_SGD())


def test_callable_model_is_steerable():
    assert compello.is_steerable(lambda x: x) is True


def test_preflight_flags_unsupported_model():
    report = compello.preflight([], model=_FakeRF())
    assert not report.ok


# --- custom assertion types (3) --------------------------------------------

def test_register_and_use_custom_assertion_type():
    from compello import math as cmath

    def my_penalty(tensor, *, cap):
        return cmath.mean(cmath.relu(cmath.asarray(tensor) - cap))

    compello.register_assertion_type("below_cap", my_penalty)
    assert "below_cap" in compello.registered_assertion_types()
    t = compello.wrap(np.array([1.0, 5.0]))
    a = expect(t, assertion_type="below_cap", name="cap", cap=2.0)
    # mean(relu([1-2, 5-2])) = mean(0, 3) = 1.5
    assert a.violation_scalar(cap=2.0) == pytest.approx(1.5)


def test_cannot_override_builtin_type():
    with pytest.raises(ValueError):
        compello.register_assertion_type("range", lambda t: t)


# --- modality relaxations (4) ----------------------------------------------

def test_soft_iou_penalty():
    from compello import math as cmath
    m = np.array([1.0, 1.0, 0.0, 0.0])
    # perfect overlap -> IoU 1.0 -> penalty 0
    assert cmath.to_float(compello.soft_iou_penalty(m, m, target=0.7)) == pytest.approx(0.0)
    # disjoint -> IoU 0 -> penalty ~target
    other = np.array([0.0, 0.0, 1.0, 1.0])
    assert cmath.to_float(compello.soft_iou_penalty(m, other, target=0.7)) == pytest.approx(0.7)


def test_soft_f1_penalty_perfect():
    from compello import math as cmath
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert cmath.to_float(compello.soft_f1_penalty(probs, labels, target=0.9)) == pytest.approx(0.0)


def test_soft_rank_penalty_positive_closer_is_zero():
    from compello import math as cmath
    q = np.array([1.0, 0.0])
    pos = np.array([1.0, 0.0])           # identical -> sim 1
    negs = np.array([[0.0, 1.0], [-1.0, 0.0]])  # orthogonal / opposite
    assert cmath.to_float(compello.soft_rank_penalty(q, pos, negs, margin=0.1)) == pytest.approx(0.0)


def test_relaxations_registered_as_types():
    types = compello.registered_assertion_types()
    for name in ("soft_iou", "soft_f1", "spectral_gate", "soft_rank"):
        assert name in types


# --- reports (5) -----------------------------------------------------------

def test_sensitivity_profiler_detects_high_impact():
    prof = compello.SensitivityProfiler(high_impact_threshold=0.01)
    # task metric linearly worsens as weight grows -> high impact, clean fit
    for i in range(120):
        w = 1.0 + 0.05 * i
        prof.observe("c", weight=w, violation=0.0, task_metric=0.1 + 0.2 * w)
    rep = prof.report()
    assert rep["c"].impact == "high"
    assert rep["c"].marginal_cost == pytest.approx(0.2, abs=0.05)


def test_non_convergence_report_infeasible():
    c = Controller(ControllerConfig(tolerance=0.01, weight_ceiling=5.0, patience=5))
    c.register("hard", 1.0)
    for _ in range(200):
        c.step({"hard": 1.0})
    entries = compello.non_convergence_report(c)
    assert entries and "infeasible" in entries[0].diagnosis


# --- plateau interventions + flag (6) --------------------------------------

def test_on_plateau_emits_intervention_then_reports_infeasible():
    cfg = ControllerConfig(tolerance=0.01, weight_ceiling=2.0, patience=3, max_attempts=2,
                           on_plateau="reduce_lr")
    c = Controller(cfg)
    c.register("k", 1.0)
    actions = []
    for _ in range(100):
        r = c.step({"k": 1.0})
        actions.extend(r.interventions.values())
    assert "reduce_lr" in actions
    # after max_attempts, escalates to report_infeasible
    assert "report_infeasible" in actions


def test_aggressive_momentum_flag_plumbed():
    cfg = ControllerConfig(aggressive_momentum_correction=True)
    assert cfg.aggressive_momentum_correction is True


def test_should_stop_on_convergence():
    cfg = ControllerConfig(tolerance=0.01, patience=3, max_steps=100000)
    c = Controller(cfg)
    c.register("k", 1.0)
    last = None
    for _ in range(5):
        last = c.step({"k": 0.0})
    assert last.should_stop is True


# --- diagnostics runner (7) ------------------------------------------------

def test_diagnostics_runner_stride():
    from compello import DiagnosticsRunner
    assert DiagnosticsRunner(telemetry="silent").enabled is False
    r = DiagnosticsRunner(telemetry="compact")
    assert r.interval == 10
    assert r.should_run(0) is True
    assert r.should_run(5) is False
    assert r.should_run(10) is True
    assert DiagnosticsRunner(telemetry="verbose").should_run(3) is True


# --- distillation bridge (8) -----------------------------------------------

def test_distillation_bridge_combines_terms():
    t = compello.wrap(np.array([-1.0, 2.0]))
    pos = expect(t, "> 0", name="pos", initial_weight=2.0)
    bridge = compello.distillation_bridge(
        teacher=object(), student=object(), constraints=[pos], distill_weight=1.0,
    )
    step = bridge.combined_loss(distillation_loss=np.array(0.5))
    # distill (0.5) + weight(2.0) * violation(mean relu(0 - [-1,2]) = 0.5) = 0.5 + 1.0
    assert step.distillation == pytest.approx(0.5)
    assert step.constraint_terms["pos"] == pytest.approx(1.0)
    assert step.total == pytest.approx(1.5)


# --- insight engine (telemetry + smart insights) ---------------------------

def test_insight_engine_emits_relaxation_block_after_stability():
    from compello import InsightEngine
    cfg = ControllerConfig(tolerance=0.02, patience=5)
    c = Controller(cfg)
    c.register("k", 1.0)
    engine = InsightEngine(c, telemetry="compact", total_steps=1000)

    # violated first so there is something to recover from
    engine.observe(c.step({"k": 1.0}), loss=0.5)

    # drive to stable satisfaction -> expect a recovery/relaxation block
    saw_relax = False
    for _ in range(50):
        si = engine.observe(c.step({"k": 0.0}), loss=0.1)
        if any("Relaxing weight" in s for s in si.insights):
            saw_relax = True
    assert saw_relax


def test_insight_engine_reports_gradient_conflict():
    from compello import InsightEngine
    c = Controller(ControllerConfig())
    c.register("k", 1.0)
    engine = InsightEngine(c, telemetry="verbose")
    r = c.step({"k": 0.5})
    si = engine.observe(r, loss=1.0, grad_conflicts={"k": {"cosine": -0.9, "projected": True}})
    assert any("conflict" in s.lower() for s in si.insights)


def test_insight_engine_telemetry_line_has_loss_and_step():
    from compello import InsightEngine
    c = Controller(ControllerConfig())
    c.register("k", 1.0)
    engine = InsightEngine(c, telemetry="compact", total_steps=100)
    r = c.step({"k": 0.5})
    si = engine.observe(r, loss=3.14)
    assert "loss: 3.14" in si.telemetry
    assert "Step 0" in si.telemetry


# --- NaN/inf guard (production hardening) ----------------------------------

def test_controller_nan_violation_does_not_poison_state():
    c = Controller(ControllerConfig(tolerance=0.01))
    c.register("k", 1.0)
    c.step({"k": 0.5})           # establish a good last value
    r = c.step({"k": float("nan")})
    st = c.states["k"]
    assert np.isfinite(st.weight)
    assert np.isfinite(st.smoothed if st.smoothed is not None else 0.0)
    assert np.isfinite(r.per_constraint["k"].raw_violation)


def test_controller_inf_violation_carried_forward():
    c = Controller(ControllerConfig(tolerance=0.01))
    c.register("k", 1.0)
    c.step({"k": 0.3})
    r = c.step({"k": float("inf")})
    # inf is replaced by the last good value (0.3), not propagated
    assert r.per_constraint["k"].raw_violation == pytest.approx(0.3)
