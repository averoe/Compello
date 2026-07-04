import numpy as np
import pytest

import compello
from compello import expect, load_config, preflight, validate
from compello.exceptions import (
    CompiledOptimizerStepConflictError,
    DistributedConfigError,
)
from compello.validation import dry_run

YAML = """
constraints:
  - name: positivity
    type: range
    target: model.output
    condition: "> 0"
    initial_weight: 1.0
  - name: token_confidence
    type: probability_floor
    target: logits[target_token]
    condition: "> 0.6"
    respect_loss_mask: true
controller:
  strategy: adaptive_pid
  patience: 500
  max_steps: 50000
  ema_decay: 0.97
  weight_ceiling: 25.0
backend: huggingface_trainer
distributed: auto
modality: text
diagnostics:
  conflict_check: true
  gradient_surgery: true
"""


def test_load_config_parses_constraints_and_controller():
    cfg = compello.CompelloConfig  # ensure exported
    parsed = load_config(YAML)
    assert len(parsed.constraints) == 2
    assert parsed.constraints[0].name == "positivity"
    assert parsed.constraints[1].respect_loss_mask is True
    assert parsed.controller.strategy == "adaptive_pid"
    assert parsed.controller.weight_ceiling == 25.0
    assert parsed.backend == "huggingface_trainer"
    assert parsed.modality == "text"


def test_load_config_rejects_unknown_backend():
    with pytest.raises(ValueError):
        load_config({"backend": "quantum", "controller": {}})


def test_load_config_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        load_config({"controller": {"strategy": "magic"}})


def test_preflight_flags_jax_distributed_auto():
    cfg = {"backend": "jax_native", "distributed": "auto"}
    report = preflight([], cfg)
    assert not report.ok
    assert any("distributed" in e for e in report.errors)
    with pytest.raises(DistributedConfigError):
        preflight([], cfg, raise_on_error=True)


def test_preflight_flags_compiled_optimizer_step():
    cfg = {"backend": "raw_pytorch", "compiled_optimizer_step": True,
           "diagnostics": {"gradient_surgery": True}}
    with pytest.raises(CompiledOptimizerStepConflictError):
        preflight([], cfg, raise_on_error=True)


def test_preflight_detects_conflicts():
    t = compello.wrap(np.array([1.0]))
    a = expect(t, "> 0.6", name="a")
    b = expect(t, "< 0.3", name="b")
    report = preflight([a, b])
    assert not report.ok


def test_validate_holdout_with_model():
    class M:
        def __call__(self, x):
            return x

    m = compello.wrap(M())
    pos = expect(m.output, "> 0", name="pos")
    data = [np.array([1.0, 2.0]), np.array([-1.0, 3.0])]
    report = validate(m, data, constraints=[pos], tolerance=1e-6)
    assert report.per_constraint["pos"].n == 2
    # first batch satisfied, second violated -> 50%
    assert report.per_constraint["pos"].satisfied_fraction == pytest.approx(0.5)


def test_dry_run_feasibility():
    def step_fn(i):
        return {"k": max(0.0, 1.0 - 0.1 * i)}
    res = dry_run(step_fn, ["k"], steps=20)
    assert res.feasible is True

    def flat(i):
        return {"k": 1.0}
    assert dry_run(flat, ["k"], steps=10).feasible is False
