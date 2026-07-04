"""Real TensorFlow / Keras 3 integration tests for the tf adapter paths.

Skipped when tensorflow is not installed. Validates the tf math backend, the
Keras optimizer beta_1 readout, momentum-buffer projection over Keras optimizer
slot variables, the ConstraintTape gradient-surgery path, and the keras.ops
math backend.
"""

import os

import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

import compello
from compello import expect
from compello.backends import tf_backend as tfb
from compello.backends.tf_backend import TFAdapter, TFBackend
from compello.controller import Controller, ControllerConfig


def test_tf_backend_registered_and_math():
    from compello import math as cmath
    assert "tensorflow" in cmath.available_backends()
    x = tf.constant([-1.0, 2.0])
    assert bool(tf.reduce_all(cmath.relu(x) == tf.constant([0.0, 2.0])))
    assert cmath.to_float(tf.constant(3.5)) == pytest.approx(3.5)


def test_expect_on_tf_tensors():
    out = compello.wrap(tf.constant([-1.0, 2.0, 3.0]))
    a = expect(out, "> 0", name="pos")
    assert a.violation_scalar() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_optimizer_beta1_from_keras_adam():
    opt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9)
    adapter = TFAdapter(optimizer=opt)
    assert adapter.optimizer_beta1() == pytest.approx(0.9)


def test_project_momentum_buffers_keras_adam():
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(4,))])
    opt = keras.optimizers.Adam(0.01)
    x = tf.random.normal((8, 4))
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))   # populate momentum
    adapter = TFAdapter(model, opt)
    task_grads = {v.path: tf.ones_like(v) for v in model.trainable_variables}
    modified = adapter.project_momentum_buffers(task_grads)
    assert modified >= 0                                          # runs without error


def test_constraint_tape_steer_gradients():
    model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
    wm = compello.wrap(model)
    pos = expect(wm.output, "> 0", name="pos")
    ctrl = Controller(ControllerConfig())
    ctrl.register_assertions([pos])
    adapter = TFAdapter(model, keras.optimizers.SGD(0.1))

    x = tf.random.normal((8, 4))
    y = tf.random.normal((8, 3))
    with tfb.ConstraintTape(controller=ctrl, assertions=[pos], adapter=adapter,
                            persistent=True) as tape:
        preds = wm(x, training=True)
        task_loss = tf.reduce_mean((preds - y) ** 2)
    steered = tape.steer_gradients(primary_loss=task_loss,
                                   variables=model.trainable_variables)
    assert len(steered) == len(model.trainable_variables)
    for g in steered:
        assert g is not None
        assert bool(tf.reduce_all(tf.math.is_finite(g)))

    # the constraint must actually be traced and contribute (not a silent no-op):
    # its own gradient must be non-None for at least one variable.
    c_grads = tape.gradient(tape._constraint_loss, model.trainable_variables)
    assert any(g is not None for g in c_grads)
    # and the steered gradient must differ from the pure task gradient
    task_only = tape.gradient(task_loss, model.trainable_variables)
    diff = sum(float(tf.reduce_sum(tf.abs(s - t)))
               for s, t in zip(steered, task_only) if s is not None and t is not None)
    assert diff > 0.0


def test_keras_ops_backend_penalty():
    from compello import math as cmath
    from compello import penalties as P
    cmath.set_backend("keras")
    try:
        val = P.hinge_range(keras.ops.convert_to_tensor([-1.0, 2.0]), ">", 0.0)
        assert cmath.to_float(val) == pytest.approx(0.5)   # mean(relu([1,0]))
    finally:
        cmath.set_backend("numpy")


def test_keras_callback_constructs():
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(2,))])
    wm = compello.wrap(model)
    pos = expect(wm.output, "> 0", name="pos")
    ctrl = Controller(ControllerConfig())
    cb = tfb.CompelloKerasCallback(ctrl, [pos], TFAdapter(model))
    assert cb.controller is ctrl
