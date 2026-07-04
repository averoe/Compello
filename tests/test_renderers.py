"""Tests for the rich terminal renderers (pre-flight shield, insight blocks,
capacity report)."""

import numpy as np
import pytest

import compello
from compello import expect, render_capacity_report, render_preflight_shield
from compello.controller import Controller, ControllerConfig
from compello.insights import InsightEngine
from compello.reports import SensitivityProfiler
from compello.report_style import Style
from compello.trainlint import lint_source


ASCII = Style(unicode=False)


# --- style -----------------------------------------------------------------

def test_style_ascii_vs_unicode_glyphs():
    assert Style(unicode=False).g("green") == "[OK]"
    assert Style(unicode=True).g("green") != "[OK]"


def test_style_status_glyph():
    s = Style(unicode=False)
    assert s.status_glyph(2, 2) == "[OK]"
    assert s.status_glyph(0, 2) == "[X]"
    assert s.status_glyph(1, 2) == "[!]"


# --- pre-flight shield -----------------------------------------------------

def test_shield_renders_why_and_fix_for_compiled_optimizer():
    src = (
        "import torch\n"
        "def train_step(batch):\n"
        "    out = model(batch)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "compiled = torch.compile(train_step)\n"
    )
    issues = lint_source(src)
    out = render_preflight_shield(issues, script="f.py", style=ASCII)
    assert "PRE-FLIGHT STATIC SHIELD" in out
    assert "CompiledOptimizerStepConflictError" in out
    assert "WHY THIS WILL HURT YOUR RUN" in out
    assert "ACTIONABLE FIX" in out
    assert "optimizer.step() stays eager" in out
    assert "PRE-FLIGHT FAILED" in out


def test_shield_clean_pass():
    src = (
        "import torch\n"
        "def train(model, loader, optimizer):\n"
        "    model.train()\n"
        "    for x, y in loader:\n"
        "        optimizer.zero_grad()\n"
        "        loss = loss_fn(model(x), y)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
    )
    out = render_preflight_shield(lint_source(src), script="f.py", style=ASCII)
    assert "ALL PRE-FLIGHT CHECKS PASSED" in out


def test_shield_from_preflight_errors():
    out = render_preflight_shield(
        [], preflight_errors=["distributed: auto invalid under jax_native ..."],
        script="cfg", style=ASCII,
    )
    assert "CONFIGURATION ERROR DETECTED" in out


# --- insight blocks --------------------------------------------------------

def _engine(**kw):
    c = Controller(ControllerConfig(tolerance=0.05, patience=5, **kw.pop("cfg", {})))
    c.register("k", 1.0)
    return c, InsightEngine(c, telemetry="compact", total_steps=100, diagnostics_interval=1,
                            modality="vision", style=ASCII, **kw)


def test_insight_block_gradient_conflict_with_layer():
    c, eng = _engine(targets={"k": "seg_mask"})
    r = c.step({"k": 0.5})
    si = eng.observe(r, loss=0.1, grad_conflicts={"k": {
        "cosine": -0.9, "projected": True, "layer": "model.layers.28.mlp",
        "task_loss": "dice_loss"}})
    text = si.render()
    assert "Runtime Insight" in text
    assert "model.layers.28.mlp" in text
    assert "Gradient Surgery active" in text
    assert "cosine similarity: -0.90" in text


def test_insight_block_momentum_grace_after_engage():
    c, eng = _engine()
    c.engage_surgery("k", beta1=0.9)
    r = c.step({"k": 0.5})
    si = eng.observe(r, loss=0.1, grad_conflicts={"k": {"cosine": -0.8, "projected": True}})
    assert any("Momentum Grace Window" in b for b in si.insights)


def test_insight_block_vanishing_zone_alpha_tune():
    c = Controller(ControllerConfig(tolerance=0.05, sharpness_g_floor=1e-4))
    c.register("p", 1.0)
    c.states["p"].alpha = 20.0
    eng = InsightEngine(c, telemetry="compact", diagnostics_interval=1, style=ASCII)
    r = c.step({"p": 0.5}, proxy_grad_norms={"p": 1e-7}, metric_satisfied={"p": False})
    si = eng.observe(r, loss=0.1)
    text = si.render()
    assert "Alpha auto-tuning triggered" in text
    assert "Hysteresis dead-band" in text


def test_insight_block_recovery_and_relaxation():
    c, eng = _engine()
    c.step({"k": 1.0})           # violated first
    eng.observe(c.step({"k": 1.0}), loss=0.5)
    saw = False
    for _ in range(40):
        si = eng.observe(c.step({"k": 0.0}), loss=0.1)
        if any("Relaxing weight" in b for b in si.insights):
            saw = True
    assert saw


def test_status_line_stays_quiet_when_stable():
    c, eng = _engine()
    # a satisfied step with no transition should produce no insight blocks
    for _ in range(10):
        si = eng.observe(c.step({"k": 0.0}), loss=0.1)
    assert si.insights == []
    assert "Bounds Compliant" in si.telemetry


# --- capacity report -------------------------------------------------------

def test_capacity_report_high_impact_and_recommendations():
    c = Controller(ControllerConfig(tolerance=0.02, weight_ceiling=1e9))
    c.register("fairness", 1.0)
    prof = SensitivityProfiler(high_impact_threshold=0.01)
    # scripted: task metric worsens linearly with weight -> clean high-impact fit
    for i in range(120):
        r = c.step({"fairness": 0.5})
        w = c.states["fairness"].weight
        prof.observe("fairness", weight=w, violation=0.5, task_metric=0.1 + 0.3 * w)
    out = render_capacity_report(c, prof, converged=False,
                                 primitive_labels={"fairness": "Fairness Primitive"},
                                 style=ASCII)
    assert "CAPACITY & SENSITIVITY REPORT" in out
    assert "Fairness Primitive" in out
    assert "Quantified Trade-off" in out
    assert "OLS R^2" in out
    assert "STRATEGIC RECOMMENDATIONS" in out
    assert "distillation_bridge" in out


def test_capacity_report_converged_low_impact():
    c = Controller(ControllerConfig(tolerance=0.02))
    c.register("cheap", 1.0)
    prof = SensitivityProfiler(high_impact_threshold=100.0)  # nothing is "high"
    for _ in range(60):
        c.step({"cheap": 0.0})
        prof.observe("cheap", weight=1.0, violation=0.0, task_metric=0.5)
    out = render_capacity_report(c, prof, converged=True, style=ASCII)
    assert "Optimization complete" in out
