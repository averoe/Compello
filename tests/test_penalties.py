import numpy as np
import pytest

from compello import math as cmath
from compello import penalties as P


def test_parse_condition_string():
    assert P.parse_condition("> 0") == (">", 0.0)
    assert P.parse_condition(">= 0.6") == (">=", 0.6)
    assert P.parse_condition("< -1.5") == ("<", -1.5)


def test_parse_condition_tuple():
    assert P.parse_condition((">", 0.5)) == (">", 0.5)


def test_parse_condition_lambda_probe():
    assert P.parse_condition(lambda y: y > 0) == (">", 0.0)
    assert P.parse_condition(lambda p: p > 0.6) == (">", 0.6)
    assert P.parse_condition(lambda x: x < 3) == ("<", 3.0)


def test_hinge_range_zero_when_satisfied():
    v = P.hinge_range(np.array([1.0, 2.0, 3.0]), ">", 0.0)
    assert cmath.to_float(v) == pytest.approx(0.0)


def test_hinge_range_positive_when_violated():
    v = P.hinge_range(np.array([-1.0, -3.0]), ">", 0.0)
    assert cmath.to_float(v) == pytest.approx(2.0)  # mean(relu(0 - x)) = mean(1,3)


def test_monotonicity_penalises_decrease():
    inc = np.array([1.0, 2.0, 3.0])
    dec = np.array([3.0, 2.0, 1.0])
    assert cmath.to_float(P.monotonicity(inc, increasing=True)) == pytest.approx(0.0)
    assert cmath.to_float(P.monotonicity(dec, increasing=True)) > 0.0


def test_invariance_zero_when_identical():
    a = np.array([1.0, 2.0])
    assert cmath.to_float(P.invariance_l2(a, a)) == pytest.approx(0.0)
    assert cmath.to_float(P.invariance_l2(a, a + 1.0)) == pytest.approx(1.0)


def test_probability_floor_masking():
    # two positions, only first is a real target
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    mask = np.array([1.0, 0.0])
    # index 0 has high prob at position 0 (satisfied) -> masked violation ~0
    v = P.probability_floor(logits, 0, ">", 0.6, mask=mask)
    assert cmath.to_float(v) == pytest.approx(0.0, abs=1e-3)


def test_probability_floor_violation_without_mask():
    logits = np.array([0.0, 0.0, 0.0])  # uniform -> p=1/3 for index 0
    v = P.probability_floor(logits, 0, ">", 0.6)
    assert cmath.to_float(v) == pytest.approx(0.6 - 1.0 / 3.0, abs=1e-6)


def test_parity_difference():
    a = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    assert cmath.to_float(P.parity(a, b)) == pytest.approx(1.0)


def test_lipschitz_penalises_excess_norm():
    assert cmath.to_float(P.lipschitz(np.array([0.5, 0.8]), 1.0)) == pytest.approx(0.0)
    assert cmath.to_float(P.lipschitz(np.array([2.0]), 1.0)) == pytest.approx(1.0)


def test_non_monotone_predicate_rejected():
    with pytest.raises(ValueError):
        P.parse_condition(lambda x: abs(x) > 1)
