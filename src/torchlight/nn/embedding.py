"""
``Embedding``: a trainable lookup table of row vectors.

    y = weight[index]

Rows are picked with :func:`torchlight.nn.functional.embedding` (an
:class:`~torchlight.autograd.indexing.IndexSelect`), so the gradient of any
loss flows back to exactly the rows that the forward pass touched.
"""

from __future__ import annotations

from typing import Optional

from ..tensor import Tensor, empty
from . import functional as F
from .module import Module, Parameter


class Embedding(Module):
    r"""
    A trainable lookup table mapping indices to dense vectors.

    Args:
        num_embeddings: size of the vocabulary (first weight dimension).
        embedding_dim: dimension of each embedded vector.
        padding_idx: if given, rows with this index are pinned to zero
            vectors whose gradient is never updated (torch semantics).

    Example:
        >>> emb = Embedding(10, 4)
        >>> emb(tensor([0, 3, 7])).shape
        (3, 4)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding num_embeddings and embedding_dim must be > 0")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(empty((num_embeddings, embedding_dim)), name="weight")
        # PyTorch's default: standard-normal rows.
        self.weight.value.normal_()
        if padding_idx is not None:
            if not 0 <= padding_idx < num_embeddings:
                raise ValueError(f"padding_idx {padding_idx} out of range [0, {num_embeddings})")
            row = self.weight.value.data_
            row[padding_idx * embedding_dim : (padding_idx + 1) * embedding_dim] = 0.0

    def forward(self, input: Tensor) -> Tensor:
        """Map each index in ``input`` to its embedding row."""
        return F.embedding(input, self.weight.value, self.padding_idx)

    def __repr__(self) -> str:
        return (
            f"Embedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
            f", padding_idx={self.padding_idx})"
        )