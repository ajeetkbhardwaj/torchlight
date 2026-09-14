"""
Backends: where tensors actually compute.

Torchlight ships three backends built on the same tiny primitive interface
(``map`` / ``zip`` / ``reduce`` / ``matmul``):

* :class:`CPUBackend` -- vectorised numpy on the CPU (the default).
* :class:`NumbaBackend` -- numba JIT kernels on the CPU (device ``"numba"``);
  transparently accelerated, same semantics.
* :class:`CudaBackend` -- kernels on an NVIDIA GPU (device ``"cuda"``);
  mirrors the CUDA sources in ``torchlight/cuda_kernels/`` (LeetGPU-verified)
  and requires a real CUDA device.  Fused ops run the compiled CUDA C kernels
  via ctypes whenever their ``.so`` files are present, otherwise numba-cuda.

:func:`get_backend` resolves a device/string into one of those singletons
(lazily imported, so vanilla installs without numba stay fast and functional).
"""

from .cpu import ARRAY_MAP, ARRAY_REDUCE, ARRAY_ZIP, CPUBackend, default_backend
from ..core.device import Device, cpu as cpu_device

__all__ = [
    "CPUBackend",
    "default_backend",
    "cpu",
    "ARRAY_MAP",
    "ARRAY_ZIP",
    "ARRAY_REDUCE",
    "get_backend",
    "NumbaBackend",
    "CudaBackend",
]


def __getattr__(name):
    """Lazily import the numba/CUDA backends so plain ``import torchlight``
    stays fast and never drags numba in until actually requested."""
    if name in ("NumbaBackend", "CudaBackend"):
        from .cuda import NumbaBackend, CudaBackend

        return NumbaBackend if name == "NumbaBackend" else CudaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: In-memory resolvable backends (no import cost, no numba requirement).
_BACKENDS = {
    "cpu": "default_backend",
    "": "default_backend",
}


def get_backend(device=None):
    """
    Resolve a device hint into a backend instance.

    ``device`` may be ``None`` (owning backend), a :class:`Device`, a string
    (``"cpu"``, ``"cuda"``), or an already-instantiated backend. The default
    backend honours the ``TORCHLIGHT_DEVICE`` environment variable.
    (``"numba"`` is available internally for kernel validation.)
    """
    if device is None:
        from ..core.device import resolve_device

        device = resolve_device(None)
    if isinstance(device, Device):
        name = device.type
    elif isinstance(device, str):
        name = device.split(":")[0].lower()
    elif all(hasattr(device, m) for m in ("map", "zip", "reduce", "matmul")):
        # Already a backend instance (has the primitive surface).
        return device
    else:
        raise TypeError(f"Cannot interpret {device!r} as a device")
    if name == "cpu":
        return default_backend
    if name == "numba":
        from .cuda import numba_backend

        return numba_backend()
    if name == "cuda":
        from .cuda import cuda_backend

        return cuda_backend()
    raise ValueError(
        f"Unknown device '{device}'.  Available: cpu, cuda"
    )