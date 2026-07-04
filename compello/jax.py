"""compello.jax -- native JAX entry point (Section 10.4).

This is the documented import path from the plan:

    import compello.jax as cjax
    state = cjax.init_controller_state(constraints=my_constraints)
    steered_grads, state = cjax.steer_step(grads, preds, state, constraints=..., axis_name="batch")

It re-exports the JAX adapter surface from ``compello.backends.jax_backend``
(which imports ``jax`` lazily and raises a clear error if it is absent) plus the
shared declaration API.
"""

from __future__ import annotations

from .assertions import expect
from .backends.jax_backend import (
    JaxBackend,
    check_distributed_config,
    init_controller_state,
    steer_step,
)
from .proxy import unwrap, wrap

__all__ = [
    "init_controller_state",
    "steer_step",
    "check_distributed_config",
    "JaxBackend",
    "expect",
    "wrap",
    "unwrap",
]
