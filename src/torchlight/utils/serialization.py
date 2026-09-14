"""
Model serialization helpers.

Two granularities are supported:

* *State dicts* -- a ``{name: numpy array}`` map of a model's parameters,
  saved as a single ``.npz`` archive (portable, torch-compatible-ish).
* *Whole models* -- pickle a model object (only safe when the model is
  pickle-able, which Torchlight modules are).

See ``examples/persistence.py`` for end-to-end usage.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict

import numpy as np

from ..nn.module import Module
from ..nn import Parameter
from ..tensor import Tensor


def state_dict(module: Module) -> Dict[str, np.ndarray]:
    """
    Collect every parameter of ``module`` into a ``{name: numpy}`` dict.

    Names look like ``"fc.weight"`` etc., matching
    :meth:`Module.named_parameters`.
    """
    out: Dict[str, np.ndarray] = {}
    for name, param in module.named_parameters():
        out[name] = param.value.to_numpy()
    return out


def load_state_dict(module: Module, state: Dict[str, np.ndarray], strict: bool = True) -> None:
    """
    Write array values from ``state`` back into ``module``'s parameters.

    Args:
        module: the model to update.
        state: a dict like :func:`state_dict` returns.
        strict: raise if key sets don't match exactly.
    """
    params = dict(module.named_parameters())
    if strict:
        missing = set(params) - set(state)
        extra = set(state) - set(params)
        if missing or extra:
            raise ValueError(
                f"state_dict mismatch: missing {sorted(missing)}, extra {sorted(extra)}"
            )
    for name, arr in state.items():
        if name not in params:
            continue
        param: Parameter = params[name]
        param.update(Tensor.make(np.asarray(arr, dtype=np.float32).reshape(-1), tuple(np.asarray(arr).shape)))


def save_state(module: Module, path: str) -> None:
    """Save ``module``'s parameters as a ``.npz`` archive at ``path``."""
    sd = state_dict(module)
    # np.savez needs a dict of arrays; store shapes alongside for safety.
    payload = {**sd}
    np.savez_compressed(path, **payload)


def load_state(module: Module, path: str) -> None:
    """Load parameters saved by :func:`save_state` into ``module``."""
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    load_state_dict(module, state)


def save_model(model: Any, path: str) -> None:
    """Persist a whole model object (or any object) with pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    """Restore an object saved by :func:`save_model`."""
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = ["state_dict", "load_state_dict", "save_state", "load_state", "save_model", "load_model"]