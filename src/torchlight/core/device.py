"""
Device abstraction.

Torchlight computes through backends: numpy on the CPU (``"cpu"``), numba
CPU-JIT (``"numba"``), and numba-cuda on NVIDIA GPUs (``"cuda"``).  A
:class:`Device` is a lightweight tag describing where a tensor computes --
the backend layer maps a device to the singleton that runs its kernels.
"""

from __future__ import annotations

import os
from typing import Optional


class Device:
    """A lightweight description of where a tensor lives / computes."""

    __slots__ = ("type", "index")

    def __init__(self, type: str, index: int = 0):
        #: Device kind: ``"cpu"`` or ``"cuda"``.
        self.type = type
        #: Device index for multi-device setups (unused today).
        self.index = index

    # -- Derived helpers ----------------------------------------------------
    @property
    def is_cpu(self) -> bool:
        return self.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"

    # -- Dunder ------------------------------------------------------------
    def __repr__(self) -> str:
        return f"device('{self.type}:{self.index}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and (self.type, self.index) == (
            other.type,
            other.index,
        )

    def __hash__(self) -> int:
        return hash((self.type, self.index))


# -- Singletons ------------------------------------------------------------
#: The CPU device.
cpu = Device("cpu")

#: The CUDA device (numba-cuda backend; requires an NVIDIA GPU + driver).
cuda = Device("cuda")

#: Recognised device names (in the order users should try them).
AVAILABLE = ("cpu", "cuda", "numba")


def resolve_device(device: "Optional[Device | str | None]") -> Device:
    """
    Normalize a device argument into a :class:`Device`.

    Accepts ``None`` (→ default device), a string like ``"cuda"``, or an
    existing ``Device``.  The default device is read from the
    ``TORCHLIGHT_DEVICE`` environment variable, falling back to ``"cpu"``.
    """
    if device is None:
        name = os.environ.get("TORCHLIGHT_DEVICE", "cpu").split(":")[0].lower()
        return Device(name)
    if isinstance(device, Device):
        return device
    name = str(device).split(":")[0].lower()
    _map = {"cpu": cpu, "cuda": cuda}
    if name not in _map:
        raise ValueError(
            f"Unknown device '{device}'.  Available: cpu, cuda, numba "
            "(set TORCHLIGHT_DEVICE env var to change the default)."
        )
    return _map[name]


__all__ = ["Device", "cpu", "cuda", "AVAILABLE", "resolve_device"]