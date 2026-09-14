"""
Normalization layers: ``LayerNorm`` and ``BatchNorm1d``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

from ..tensor import Tensor, ones, zeros
from . import functional as F, init as _init
from .module import Module, Parameter


class LayerNorm(Module):
    """
    Layer normalisation over the trailing ``normalized_shape`` dims.

    Normalisation statistics are per-sample, so behaviour is identical in
    train and eval mode.  Optional learnable affine ``weight/``bias``.
    """

    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.affine = affine

        if affine:
            shape = (int(self.normalized_shape[0]),) if len(self.normalized_shape) == 1 else tuple(self.normalized_shape)
            self.weight = Parameter(ones(shape), name="weight")
            self.bias = Parameter(zeros(shape), name="bias")
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        out = F.layer_norm(input, self.normalized_shape, self.eps)
        if self.weight is not None:
            out = out * self.weight.value
        if self.bias is not None:
            out = out + self.bias.value
        return out

    def __repr__(self) -> str:
        return (
            f"LayerNorm({self.normalized_shape}, eps={self.eps}, "
            f"affine={self.affine})"
        )


class LayerNorm1d(LayerNorm):
    """
    Layer normalisation for a mini-batch of *1D inputs* -- i.e. tensors of
    shape ``(batch, dim)`` with ``dim`` learnable gain/bias parameters.

    This is the ``LayerNorm1d`` layer of the reference implementation: a thin
    2D-input wrapper around :class:`LayerNorm` with ``normalized_shape=(dim,)``.
    """

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__(normalized_shape=(dim,), eps=eps, affine=affine)

    def __repr__(self) -> str:
        return (
            f"LayerNorm1d({int(self.normalized_shape[0])}, eps={self.eps}, "
            f"affine={self.affine})"
        )


class BatchNorm1d(Module):
    """
    Batch normalisation for ``(N, C)`` (or ``(N, C, L)``) inputs.

    While training, normalises using each batch's statistics and updates
    running buffers (``running_mean`` / ``running_var``) with momentum;
    while evaluating, normalises using the running buffers.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Running statistics (not trainable, not part of the gradient graph).
        self.register_buffer("running_mean", zeros((num_features,)))
        self.register_buffer("running_var", ones((num_features,)))

        if affine:
            self.weight = Parameter(ones((num_features,)), name="weight")
            self.bias = Parameter(zeros((num_features,)), name="bias")
        else:
            self.weight = None
            self.bias = None

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Store a non-trainable tensor under ``name`` (plain attribute)."""
        object.__setattr__(self, name, tensor)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm1d expects {self.num_features} channels, got {input.shape[1]}")
        n = input.shape[0]

        if self.training:
            # Batch statistics over the batch dim (kept as size 1 to broadcast).
            batch_mean = input.mean(dim=0)
            batch_var = input.var(dim=0)

            # Update running averages (momentum EMA on float data).
            rm = self.running_mean
            rv = self.running_var
            mean_arr = batch_mean.to_numpy().reshape(-1)
            var_arr = batch_var.to_numpy().reshape(-1)
            rm.data_[:] = ((1.0 - self.momentum) * rm.data_ + self.momentum * mean_arr)
            rv.data_[:] = ((1.0 - self.momentum) * rv.data_ + self.momentum * var_arr)

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        xhat = (input - mean) / (var + self.eps).sqrt()
        if self.weight is not None:
            xhat = xhat * self.weight.value
        if self.bias is not None:
            xhat = xhat + self.bias.value
        return xhat

    def __repr__(self) -> str:
        return (
            f"BatchNorm1d({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )