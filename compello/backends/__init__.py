"""Optional deep-learning backend adapters (Sections 3.3, 4, 10).

None of these modules are imported by ``compello`` at package import time. Each
imports its framework lazily and only when the user explicitly imports the
adapter (e.g. ``import compello.backends.torch``). If the framework is not
installed, importing the adapter raises a clear ``BackendNotAvailableError``.

This is the optional-backend pattern from Section 9: a single-backend install
(say, JAX-only) never imports torch or tensorflow.

Importantly, these adapters are the parts of the spec that *cannot* be exercised
without the corresponding framework installed. The framework-independent core
(assertion DSL, controller math, diagnostics, trainlint) is fully functional and
tested against the numpy reference backend regardless of whether any of these
are importable.
"""

from __future__ import annotations

from typing import List


def available() -> List[str]:
    """Return the deep-learning backends importable in this environment."""
    found: List[str] = []
    for name, module in (("torch", "torch"), ("tensorflow", "tensorflow"), ("jax", "jax")):
        try:
            __import__(module)
            found.append(name)
        except Exception:
            pass
    return found


__all__ = ["available"]
