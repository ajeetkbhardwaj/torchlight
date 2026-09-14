"""
torchlight.core -- the raw foundation layer.

* :mod:`device`       -- device abstraction (cpu / cuda / numba)
* :mod:`tensor_data`  -- storage + shape/strides (numpy float32 host buffer)
* :mod:`ops`          -- scalar mathematical operators (the op spec)
"""

from . import device, ops, tensor_data  # noqa: F401
from .device import Device, cpu, cuda, resolve_device
from .tensor_data import (
    Storage,
    TensorData,
    UserIndex,
    UserShape,
    UserStrides,
    shape_broadcast,
    strides_from_shape,
)

__all__ = [
    "Device",
    "cpu",
    "cuda",
    "resolve_device",
    "Storage",
    "TensorData",
    "UserIndex",
    "UserShape",
    "UserStrides",
    "shape_broadcast",
    "strides_from_shape",
    "device",
    "ops",
    "tensor_data",
]