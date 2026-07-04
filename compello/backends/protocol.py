"""The training-loop integration protocol (Section 3.3).

This is distinct from ``compello.math.Backend`` (which is only array ops). An
*adapter* wires the controller + assertions into a concrete training loop and
provides the framework-specific safety mechanisms of Section 4 that have no
portable form: distributed collective sync (4.1), backward-side gradient access
for surgery (4.6), and reading the optimizer's first-moment decay (4.7).

Each concrete adapter (torch/tf/jax) implements this shape. The methods that
differ fundamentally by framework are called out in each adapter's docstring
with the specific guarantee difference (per the plan's "document the limits"
principle).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TrainingAdapter(Protocol):
    name: str

    def sync_violations(self, local_violations: Dict[str, float]) -> Dict[str, float]:
        """Batched, one-collective-per-step cross-replica average (4.1/4.1b).

        Returns globally-averaged violations. On a single device this is the
        identity. The batching discipline (one collective covering all
        constraints) is preserved on every backend even though the collective
        API differs (all_reduce / strategy.reduce / lax.pmean).
        """
        ...

    def read_gradients(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Return materialised parameter gradients (post-backward, pre-step).

        ``scope`` implements ``gradient_surgery_scope`` (5.6): e.g.
        ``"last_n_layers:8"`` restricts which parameters are returned so the
        O(N) cosine cost is bounded.
        """
        ...

    def write_gradients(self, grads: Dict[str, Any]) -> None:
        """Write surgery-corrected gradients back before the optimizer step."""
        ...

    def optimizer_beta1(self) -> Optional[float]:
        """First-moment decay read once from optimizer config (4.7), not live
        state. None if the optimizer has no first-moment term."""
        ...
