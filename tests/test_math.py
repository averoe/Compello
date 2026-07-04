import numpy as np
import pytest

from compello import math as cmath


def test_numpy_backend_registered():
    assert "numpy" in cmath.available_backends()


def test_relu_and_sigmoid():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(cmath.relu(x), [0.0, 0.0, 3.0])
    assert np.allclose(cmath.sigmoid(np.array([0.0])), [0.5])


def test_sigmoid_numerically_stable_large_inputs():
    # must not overflow for large-magnitude inputs
    out = cmath.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(1.0, abs=1e-9)


def test_softmax_sums_to_one():
    s = cmath.softmax(np.array([1.0, 2.0, 3.0]))
    assert s.sum() == pytest.approx(1.0)


def test_to_float_scalar_and_vector():
    assert cmath.to_float(np.array(2.5)) == pytest.approx(2.5)
    assert cmath.to_float(np.array([2.0, 4.0])) == pytest.approx(3.0)


def test_norm_and_dot():
    a = np.array([3.0, 4.0])
    assert cmath.to_float(cmath.norm(a)) == pytest.approx(5.0)
    assert cmath.to_float(cmath.dot(a, a)) == pytest.approx(25.0)


def test_unavailable_backend_raises():
    with pytest.raises(Exception):
        cmath.get_backend("does-not-exist")
