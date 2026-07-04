"""Controller-state checkpoint serialization (Section 6.2).

Compello checkpoints the controller-internal state Cooper doesn't have -- slow/
fast EMA buffers, the rolling-baseline accumulator, stability + plateau
counters, and per-constraint adaptive sharpness with its hysteresis arm -- so a
resumed run does not restart the smoothing layer cold and momentarily
reintroduce the derivative-kick risk (4.2) it exists to prevent.

The serialized payload is a plain, JSON-safe dict (``Controller.to_dict``), so
the *format* is a transport detail chosen by file extension:

  .json                      -> portable JSON (always available; the tested path)
  .pt                        -> torch.save    (requires torch)
  .npz                       -> numpy archive (requires numpy)
  .keras / .orbax / dir      -> orbax PyTree checkpoint (requires orbax)

The controller state itself is identical across all formats; only the container
differs, which is exactly the Section 6.2 guarantee ("same controller-state
fields regardless of format").
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .controller.controller import Controller, ControllerConfig


def save_controller(controller: Controller, path: str) -> None:
    """Serialize controller state to ``path`` (format inferred from extension)."""
    payload = controller.to_dict()
    fmt = _infer_format(path)
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, allow_nan=True)
    elif fmt == "pt":
        _save_torch(payload, path)
    elif fmt == "npz":
        _save_npz(payload, path)
    elif fmt == "orbax":
        _save_orbax(payload, path)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported checkpoint format for {path!r}")


def load_controller(path: str, config: Optional[ControllerConfig] = None) -> Controller:
    """Reconstruct a Controller from a checkpoint written by ``save_controller``."""
    fmt = _infer_format(path)
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif fmt == "pt":
        payload = _load_torch(path)
    elif fmt == "npz":
        payload = _load_npz(path)
    elif fmt == "orbax":
        payload = _load_orbax(path)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported checkpoint format for {path!r}")

    controller = Controller(config or ControllerConfig())
    controller.load_dict(payload)
    return controller


def _infer_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".pt") or lower.endswith(".pth"):
        return "pt"
    if lower.endswith(".npz"):
        return "npz"
    if lower.endswith(".keras") or lower.endswith(".orbax") or os.path.isdir(path):
        return "orbax"
    return "json"


# -- torch (.pt) -----------------------------------------------------------

def _save_torch(payload: Dict[str, Any], path: str) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(".pt checkpoints require torch installed") from exc
    torch.save(payload, path)


def _load_torch(path: str) -> Dict[str, Any]:  # pragma: no cover - needs torch
    import torch

    return torch.load(path, weights_only=False)


# -- numpy (.npz) ----------------------------------------------------------

def _save_npz(payload: Dict[str, Any], path: str) -> None:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(".npz checkpoints require numpy installed") from exc
    # store the JSON blob as a single string entry -- keeps the nested,
    # ragged controller dict intact without flattening every field.
    np.savez(path, compello_state=json.dumps(payload))


def _load_npz(path: str) -> Dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["compello_state"]))


# -- orbax (JAX PyTree) ----------------------------------------------------

def _save_orbax(payload: Dict[str, Any], path: str) -> None:  # pragma: no cover
    try:
        import orbax.checkpoint as ocp
    except Exception as exc:
        raise RuntimeError(
            "orbax checkpoints require `orbax-checkpoint` installed (JAX backend)"
        ) from exc
    ckpt = ocp.PyTreeCheckpointer()
    ckpt.save(os.path.abspath(path), payload)


def _load_orbax(path: str) -> Dict[str, Any]:  # pragma: no cover
    import orbax.checkpoint as ocp

    ckpt = ocp.PyTreeCheckpointer()
    return ckpt.restore(os.path.abspath(path))
