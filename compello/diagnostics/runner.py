"""Diagnostic-layer overhead control (Section 5.6).

The expensive diagnostics -- gradient-surgery cosine similarity (5.2) and the
rolling-regression refit behind insight messages (5.1a) -- do not have to run
every step. ``DiagnosticsRunner`` gates them on a configurable stride
(``diagnostics_interval``) derived from the chosen telemetry verbosity:

  verbose -> every step
  compact -> every 10 steps
  silent  -> disabled

so diagnostic cost is tied to how much detail the user actually asked to see.
This centralises the "should I pay for diagnostics this step" decision so every
call site respects the same budget.
"""

from __future__ import annotations

from typing import Optional

_STRIDE_BY_VERBOSITY = {"verbose": 1, "compact": 10, "silent": 0}


class DiagnosticsRunner:
    def __init__(self, *, telemetry: str = "compact", interval: Optional[int] = None):
        self.telemetry = telemetry
        if interval is not None:
            self.interval = int(interval)
        else:
            self.interval = _STRIDE_BY_VERBOSITY.get(telemetry, 10)

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def should_run(self, step: int) -> bool:
        """True if diagnostics should run on ``step`` given the stride."""
        if not self.enabled:
            return False
        return step % self.interval == 0

    def maybe_run(self, step: int, fn, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` only on scheduled steps; else return None."""
        if self.should_run(step):
            return fn(*args, **kwargs)
        return None
