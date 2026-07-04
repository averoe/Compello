import numpy as np
import pytest

import compello
from compello import expect
from compello.diagnostics import (
    ColdStartMonitor,
    RollingRegressor,
    apply_gradient_surgery,
    cosine_similarity,
    detect_conflicts,
)
from compello.diagnostics.conflict import CONTRADICTION, TENSION
from compello.diagnostics.regression import steps_from_rate


def test_gradient_surgery_projects_conflicting_component():
    task = np.array([1.0, 0.0])
    constraint = np.array([-0.5, 1.0])
    res = apply_gradient_surgery(task, constraint)
    assert res.conflict is True and res.projected is True
    # corrected gradient must be orthogonal to the task gradient
    assert float(np.dot(res.corrected_constraint_grad, task)) == pytest.approx(0.0, abs=1e-9)


def test_gradient_surgery_noop_when_aligned():
    task = np.array([1.0, 1.0])
    constraint = np.array([1.0, 0.5])
    res = apply_gradient_surgery(task, constraint)
    assert res.conflict is False and res.projected is False


def test_cosine_similarity_bounds():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_conflict_detection_contradiction():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.6", name="a")
    b = expect(t, "< 0.3", name="b")
    reports = detect_conflicts([a, b])
    assert any(r.severity == CONTRADICTION for r in reports)


def test_conflict_detection_tension_invariance_vs_monotonicity():
    m = compello.wrap(lambda x: x)
    a = expect(m, invariant_to=lambda x: x, name="inv")
    b = expect(m, monotonic_in="age", name="mono")
    reports = detect_conflicts([a, b])
    assert any(r.severity == TENSION for r in reports)


def test_no_conflict_for_compatible_ranges():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.1", name="a")
    b = expect(t, "< 0.9", name="b")
    reports = detect_conflicts([a, b])
    assert all(r.severity != CONTRADICTION for r in reports)


def test_rolling_regressor_recovers_linear_trend():
    reg = RollingRegressor(window=100, refit_interval=10)
    v = 1.0
    for i in range(120):
        feats = [v, -0.01, i]
        v = max(0.0, v - 0.01)
        reg.observe(feats, v)
    reg.refit()
    assert reg.fit is not None
    assert reg.fit.r_squared > 0.9


def test_steps_from_rate():
    assert steps_from_rate(1.0, 0.0, -0.1) == pytest.approx(10.0)
    assert steps_from_rate(1.0, 0.0, 0.1) is None  # not improving
    assert steps_from_rate(0.0, 0.0, -0.1) == pytest.approx(0.0)


def test_regression_suppresses_low_r2(monkeypatch):
    # random-walk target -> low R^2 -> suppressed projection
    import random
    random.seed(1)
    reg = RollingRegressor(window=80, refit_interval=10)
    for i in range(100):
        reg.observe([random.random(), random.random(), i], random.random())
    reg.refit()
    steps, conf = reg.estimate_recovery_steps(1.0)
    assert conf in ("suppressed", "low")


def test_cold_start_relaxes_then_enforces():
    cs = ColdStartMonitor(window=20, soft_start_steps=50, variance_threshold=0.01)
    early = cs.update(10.0)
    assert early.relaxation < 1.0 and early.stabilised is False
    for _ in range(60):
        st = cs.update(1.0)
    assert st.stabilised is True and st.relaxation == 1.0
