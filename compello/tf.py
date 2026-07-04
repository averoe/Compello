"""compello.tf -- native TensorFlow entry point (Section 10.3).

This is the documented import path from the plan:

    from compello.tf import ConstraintTape, expect

It re-exports the TensorFlow adapter surface from ``compello.backends.tf_backend``
(which imports ``tensorflow`` lazily and raises a clear error if it is absent)
plus the shared, framework-independent ``expect``/``wrap`` declaration API.
"""

from __future__ import annotations

from .assertions import expect
from .backends.tf_backend import (
    CompelloKerasCallback,
    ConstraintTape,
    TFAdapter,
    TFBackend,
)
from .proxy import unwrap, wrap

__all__ = [
    "ConstraintTape",
    "TFAdapter",
    "TFBackend",
    "CompelloKerasCallback",
    "expect",
    "wrap",
    "unwrap",
]
