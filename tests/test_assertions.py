import numpy as np
import pytest

import compello
from compello import expect
from compello.assertions import (
    INVARIANCE,
    LIPSCHITZ,
    MONOTONICITY,
    PARITY,
    PROBABILITY_FLOOR,
    RANGE,
)
from compello.exceptions import AmbiguousAssertionError


class Lin:
    def __call__(self, x):
        return x * 2.0


def test_output_target_dispatches_to_range():
    m = compello.wrap(Lin())
    m(np.array([1.0]))
    a = expect(m.output, "> 0", name="pos")
    assert a.kind == RANGE and a.op == ">" and a.threshold == 0.0


def test_logit_target_dispatches_to_probability_floor():
    logits = compello.wrap(np.array([2.0, 1.0]))
    a = expect(logits[0], "> 0.6")
    assert a.kind == PROBABILITY_FLOOR
    assert a.params["index"] == 0


def test_keyword_dispatch():
    m = compello.wrap(Lin())
    assert expect(m, monotonic_in="age").kind == MONOTONICITY
    assert expect(m, invariant_to=lambda x: x).kind == INVARIANCE
    assert expect(m, parity_across="grp").kind == PARITY
    assert expect(m, lipschitz_bound=1.0).kind == LIPSCHITZ


def test_ambiguous_raises_without_wrap_or_type():
    with pytest.raises(AmbiguousAssertionError):
        expect(np.array([1.0]), "> 0")


def test_escape_hatch_assertion_type():
    a = expect(np.array([-1.0, 2.0]), "> 0", assertion_type="range", name="raw")
    assert a.kind == RANGE
    # resolver reads the raw tensor
    assert a.violation_scalar() == pytest.approx(0.5)  # mean(relu(0-x)) over [-1,2] = 0.5


def test_live_resolver_tracks_forward():
    m = compello.wrap(Lin())
    m(np.array([-1.0, -1.0]))
    a = expect(m.output, "> 0", name="pos")
    # currently violated (outputs -2, -2)
    assert a.violation_scalar() == pytest.approx(2.0)
    # new forward produces satisfying outputs
    m(np.array([1.0, 1.0]))
    assert a.violation_scalar() == pytest.approx(0.0)


def test_probability_floor_uses_mask_when_requested():
    logits = compello.wrap(np.array([[5.0, 0.0], [0.0, 5.0]]))
    a = expect(logits[0], "> 0.6", respect_loss_mask=True, name="conf")
    mask = np.array([1.0, 0.0])
    assert a.violation_scalar(mask=mask) == pytest.approx(0.0, abs=1e-3)


def test_two_keywords_rejected():
    m = compello.wrap(Lin())
    with pytest.raises(ValueError):
        expect(m, monotonic_in="a", parity_across="b")
