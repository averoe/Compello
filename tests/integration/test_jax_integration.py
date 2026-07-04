"""Real JAX integration tests for the jax adapter paths.

Skipped when jax is not installed. Validates the jax math backend, controller-
state PyTree, eager steer_step, the static-shape violation virtualization
(init/set/sync with jnp.where), the pmean collective under pmap, the Optax
momentum-buffer projection, and the distributed-config guard.
"""

import pytest

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")
import jax.numpy as jnp

import compello
from compello import expect
from compello.backends import jax_backend as jb
from compello.exceptions import DistributedConfigError


def test_jax_backend_registered_and_math():
    from compello import math as cmath
    assert "jax" in cmath.available_backends()
    x = jnp.array([-1.0, 2.0])
    assert jnp.allclose(cmath.relu(x), jnp.array([0.0, 2.0]))
    assert cmath.to_float(jnp.array(3.5)) == pytest.approx(3.5)
    assert jnp.allclose(cmath.softmax(jnp.array([0.0, 0.0])), jnp.array([0.5, 0.5]))


def test_expect_on_jax_arrays():
    out = compello.wrap(jnp.array([-1.0, 2.0, 3.0]))
    a = expect(out, "> 0", name="pos")
    # mean(relu(0 - x)) over [-1,2,3] = mean(1,0,0) = 1/3
    assert a.violation_scalar() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_init_controller_state_is_pytree():
    a = expect(compello.wrap(jnp.array([1.0])), "> 0", name="c")
    state = jb.init_controller_state([a])
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0
    assert "c" in state and "weight" in state["c"]


def test_steer_step_eager_threads_state():
    a = expect(compello.wrap(jnp.array([1.0])), "> 0", name="c")
    state = jb.init_controller_state([a])
    grads = {"w": jnp.ones((3,))}
    new_grads, new_state = jb.steer_step(grads, None, state, constraints=[a])
    assert "c" in new_state
    assert jnp.allclose(new_grads["w"], grads["w"])   # grads passthrough (no surgery here)


def test_violation_virtualization_static_shape():
    buf, mask = jb.init_violation_buffer(4)
    assert buf.shape == (4,) and mask.shape == (4,)
    buf, mask = jb.set_violation(buf, mask, 1, 0.7)
    buf, mask = jb.set_violation(buf, mask, 3, 0.2)
    synced = jb.sync_violations_static(buf, mask)      # no axis -> masked identity
    assert synced.shape == (4,)                        # shape never changes
    assert float(synced[0]) == 0.0 and float(synced[2]) == 0.0   # inactive rows zeroed
    assert float(synced[1]) == pytest.approx(0.7)


def test_pmean_under_pmap_single_device():
    n = jax.local_device_count()

    def f(buf, mask):
        return jb.sync_violations_static(buf, mask, axis_name="batch")

    pf = jax.pmap(f, axis_name="batch")
    buf = jnp.stack([jnp.array([0.4, 0.0, 0.0, 0.0]) for _ in range(n)])
    mask = jnp.stack([jnp.array([True, False, False, False]) for _ in range(n)])
    out = pf(buf, mask)
    assert out.shape == (n, 4)
    assert float(out[0][0]) == pytest.approx(0.4)      # active-row mean preserved


def test_project_optax_momentum_runs():
    params = {"w": jnp.ones((3,))}
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    grads = {"w": jnp.ones((3,))}
    updates, opt_state = opt.update(grads, opt_state, params)   # populate mu
    task_grads = {"w": jnp.ones((3,))}
    new_state = jb.project_optax_momentum(opt_state, task_grads)
    # returns a structurally-valid opt_state (tuple of transform states)
    assert new_state is not None


def test_distributed_auto_guard():
    with pytest.raises(DistributedConfigError):
        jb.check_distributed_config("auto", None)
    jb.check_distributed_config("off", "batch")        # explicit axis is fine
