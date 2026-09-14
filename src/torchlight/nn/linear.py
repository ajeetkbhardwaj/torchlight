"""
``Linear``: the fully-connected (dense) layer.

    y = x @ W^T + b

The weight is stored with shape ``(out_features, in_features)`` -- like
PyTorch -- so the forward pass uses ``x @ W^T``.
"""

from __future__ import annotations

from ..tensor import Tensor, zeros
from . import init as _init
from .module import Module, Parameter


class Linear(Module):
    r"""
    Applies an affine transform over the last input dimension.

    Args:
        in_features: width of each input sample.
        out_features: width of each output sample.
        bias: whether to include the learnable bias term.

    Example:
        >>> layer = Linear(4, 3)
        >>> x = tensor([[1., 2., 3., 4.]])          # (1, 4)
        >>> layer(x).shape
        (1, 3)
    """

    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight: (out, in), He-uniform initialised (the default for ReLU nets).
        self.weight = Parameter(zeros((out_features, in_features)), name="weight")
        _init.kaiming_uniform_(self.weight.value, a=0.0)

        if bias:
            # Bias: (out,); initialised in the same spirit as PyTorch.
            self.bias = Parameter(zeros((out_features,)), name="bias")
            _init.uniform_(self.bias.value, -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        """``x @ W^T + b``; input may be any shape with trailing width."""
        if input.dims < 1 or input.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear: expected trailing dim {self.in_features}, got {input.shape}"
            )
        out = input @ self.weight.value.transpose(0, 1)
        if self.bias is not None:
            # bias broadcasts across every leading (batch) dimension.
            out = out + self.bias.value
        return out

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )