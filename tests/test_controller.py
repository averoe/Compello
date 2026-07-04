import pytest

from compello.controller import (
    ADAPTIVE_PID,
    DUAL_ASCENT,
    FIXED,
    Controller,
    ControllerConfig,
    PIDGains,
    update_ema_layer,
)
from compello.controller.sharpness import update_sharpness
from compello.controller.state import ConstraintState


def test_fixed_strategy_keeps_weight_constant():
    c = Controller(ControllerConfig(strategy=FIXED))
    c.register("k", initial_weight=3.0)
    for _ in range(10):
        c.step({"k": 1.0})
    assert c.states["k"].weight == 3.0


def test_adaptive_pid_raises_weight_under_persistent_violation():
    c = Controller(ControllerConfig(strategy=ADAPTIVE_PID, tolerance=0.01))
    c.register("k", initial_weight=1.0)
    for _ in range(50):
        c.step({"k": 1.0})
    assert c.states["k"].weight > 1.0


def test_adaptive_pid_relaxes_when_stable():
    cfg = ControllerConfig(strategy=ADAPTIVE_PID, tolerance=0.05, patience=10)
    c = Controller(cfg)
    c.register("k", initial_weight=5.0)
    # first violate to build weight, then satisfy for a long stable window
    for _ in range(20):
        c.step({"k": 1.0})
    peak = c.states["k"].weight
    for _ in range(200):
        c.step({"k": 0.0})
    assert c.states["k"].weight < peak


def test_weight_ceiling_caps_and_locks():
    cfg = ControllerConfig(strategy=ADAPTIVE_PID, tolerance=0.01, weight_ceiling=5.0)
    c = Controller(cfg)
    c.register("hard", initial_weight=1.0)
    res = None
    for _ in range(500):
        res = c.step({"hard": 1.0})
    assert c.states["hard"].weight == pytest.approx(5.0)
    assert c.states["hard"].ceiling_locked is True
    assert "hard" in res.plateau_flags


def test_dual_ascent_monotonic_increase_on_violation():
    c = Controller(ControllerConfig(strategy=DUAL_ASCENT, dual_lr=0.1, weight_ceiling=1e9))
    c.register("k", initial_weight=0.0)
    prev = 0.0
    for _ in range(10):
        c.step({"k": 1.0})
        assert c.states["k"].weight >= prev
        prev = c.states["k"].weight
    assert prev == pytest.approx(1.0)  # 10 * 0.1 * 1.0


def test_converged_flag_after_patience():
    cfg = ControllerConfig(strategy=ADAPTIVE_PID, tolerance=0.01, patience=5)
    c = Controller(cfg)
    c.register("k", 1.0)
    results = [c.step({"k": 0.0}) for _ in range(6)]
    assert results[-1].converged is True


def test_checkpoint_roundtrip():
    cfg = ControllerConfig()
    c = Controller(cfg)
    c.register("a", 1.0)
    c.register("b", 2.0)
    for _ in range(30):
        c.step({"a": 0.5, "b": 0.1})
    d = c.to_dict()
    c2 = Controller(cfg)
    c2.load_dict(d)
    assert c2.to_dict() == d
    assert c2.weights == c.weights


def test_ema_spike_rejection_vs_sustained_shift():
    st = ConstraintState("k", baseline_window=50)
    kw = dict(ema_decay=0.97, ema_fast_decay=0.7, ema_override_steps=5)
    # build a calm baseline
    for _ in range(40):
        update_ema_layer(st, 0.1, **kw)
    override_after_single_spike = st.override_remaining
    # a single spike then back to calm -> should not trigger sustained override
    update_ema_layer(st, 10.0, **kw)
    update_ema_layer(st, 0.1, **kw)
    assert st.consecutive_elevated < 5
    # a sustained shift should eventually trigger the override
    for _ in range(10):
        update_ema_layer(st, 5.0, **kw)
    assert st.override_remaining > 0 or st.consecutive_elevated >= 5


def test_sharpness_hysteresis_no_chatter():
    st = ConstraintState("p")
    st.alpha = 10.0
    # gradient vanishes -> scale down and disarm
    update_sharpness(st, proxy_grad_norm=1e-6, metric_satisfied=False,
                     g_floor=1e-4, sharpness_hysteresis=1.5)
    after_down = st.alpha
    assert after_down < 10.0
    assert st.sharpness_armed is False
    # a value just above the floor but below the re-arm band must NOT scale up
    update_sharpness(st, proxy_grad_norm=1.2e-4, metric_satisfied=False,
                     g_floor=1e-4, sharpness_hysteresis=1.5)
    assert st.alpha == pytest.approx(after_down)
    assert st.sharpness_armed is False
