"""
Stateless functions over tensors (``nn.functional``).

These mirror ``torch.nn.functional``: they take tensors, return tensors, and
carry no learnable state.  Modules (``Linear``, ``ReLU``, ...) wrap them and
own parameters where relevant.

All functionals compose the primitive autograd ops, so gradients flow
automatically.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..autograd.convolutions import Conv1d, Conv2d
from ..autograd.indexing import IndexSelect
from ..tensor import Tensor, from_numpy


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def relu(input: Tensor) -> Tensor:
    """Rectified linear unit applied elementwise."""
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""
    Leaky ReLU: :math:`x` if :math:`x>0` else :math:`\alpha x`.

    Composed from ReLU as :math:`\alpha x + (1-\alpha)\,\mathrm{relu}(x)` so
    its gradient flows through the (already implemented) ReLU op.
    """
    return negative_slope * input + (1.0 - negative_slope) * input.relu()


def sigmoid(input: Tensor) -> Tensor:
    """Logistic sigmoid applied elementwise (numerically stable)."""
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Hyperbolic tangent applied elementwise."""
    return input.tanh()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    r"""
    Softmax over ``dim``.

    .. math:: \sigma(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    A numerically stable variant subtracts the max before exponentiating
    (the shift cancels in the ratio).
    """
    if dim < 0:
        dim += input.dims
    shifted = input - input.max(dim)
    e = shifted.exp()
    return e / e.sum(dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    r"""
    Log-softmax over ``dim``, using the stable log-sum-exp form:

    .. math:: \log\sigma(x)_i = x_i - \log\sum_j e^{x_j}
    """
    if dim < 0:
        dim += input.dims
    mx = input.max(dim)
    lse = (input - mx).exp().sum(dim).log() + mx
    return input - lse


def softplus(input: Tensor) -> Tensor:
    r"""
    Numerically stable :math:`\log(1 + e^x)`:

    .. math:: \mathrm{softplus}(x) = \max(x,0) + \log(1 + e^{-|x|})
    """
    pos = input.relu()                 # max(x, 0)
    neg = (-input).relu()              # -min(x, 0) = max(-x, 0)
    return pos + ((-(pos + neg)).exp() + 1.0).log()


def gelu(input: Tensor, approximate: str = "tanh") -> Tensor:
    r"""
    Gaussian error linear unit (the ``tanh`` approximation), elementwise:

    .. math::
        \mathrm{GELU}(x) = 0.5 x \left(1 + \tanh\left(
        \sqrt{\tfrac{2}{\pi}} (x + 0.044715 x^3)\right)\right)

    The approximation is differentiable through Torchlight's primitives
    (no ``erf`` op), matching the reference GELU used in the MinTorch
    assignment.
    """
    if approximate != "tanh":
        raise ValueError(f"gelu only supports approximate='tanh', got {approximate!r}")
    c = math.sqrt(2.0 / math.pi)
    inner = input + 0.044715 * input ** 3
    return 0.5 * input * (1.0 + (c * inner).tanh())


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Randomly drop elements with probability ``p`` eventwise, using *inverted
    dropout* (survivors are scaled by ``1/(1-p)`` so expectations are
    unchanged).  The identity function when ``training`` is ``False``.
    """
    if not training or p == 0.0:
        return input
    keep_prob = 1.0 - p
    keep = (input.rand_like() > p).contiguous()  # 1.0 with prob (1-p)
    return (input * keep) / keep_prob


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error between (broadcastable) ``input`` and ``target``.

    ``reduction`` is one of ``"mean"`` (default), ``"sum"`` or ``"none"``.
    """
    diff = input - target
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    return sq


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error between ``input`` and ``target``."""
    adiff = abs(input - target)
    if reduction == "mean":
        return adiff.mean()
    if reduction == "sum":
        return adiff.sum()
    return adiff


def binary_cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    r"""
    Binary cross entropy for *probability* inputs (after a sigmoid):

    .. math:: \mathcal{L} = -\frac1N\sum_i\bigl[t_i\log p_i + (1-t_i)\log(1-p_i)\bigr]

    Inputs are clamped to :math:`[\epsilon, 1-\epsilon]` to keep ``log``
    well-defined.  For raw logits use
    :func:`binary_cross_entropy_with_logits`.
    """
    eps = 1e-6
    p = input.clamp(eps, 1.0 - eps)
    one = input._ensure_tensor(1.0)
    return -(target * p.log() + (one - target) * (one - p).log()).mean()


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor) -> Tensor:
    r"""
    Numerically stable BCE for raw logits:

    .. math:: \mathcal{L} = \frac1N\sum_i \bigl[\mathrm{softplus}(x_i) - t_i x_i\bigr]
    """
    return (softplus(input) - target * input).mean()


def cross_entropy(input: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    r"""
    Cross entropy between class logits and *index* targets:

    * ``input``: logits shaped ``(..., C, ...)`` (``C`` along ``dim``).
    * ``target``: class indices, shape equal to ``input`` with ``dim`` removed.

    Computed as the gathered negative log-softmax:
    :math:`\mathcal{L} = -\frac1N\sum \log\sigma(x)_t`.
    """
    if dim < 0:
        dim += input.dims
    lsm = log_softmax(input, dim)
    return -lsm._gather_index(target, dim).mean()


def nll_loss(log_probs: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    """
    Negative log-likelihood from log-probabilities + index targets.
    """
    if dim < 0:
        dim += log_probs.dims
    return -log_probs._gather_index(target, dim).mean()


def softmax_loss(logits: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    r"""
    Softmax + cross entropy loss with *no* reduction (per-sample losses).

    ``logits`` is ``(minibatch, C)`` raw logits and ``target`` holds the
    ``minibatch`` true labels; the result is one (positive) scalar per sample
    (shape ``(minibatch,)``):

    .. math:: \mathcal{L}_i = -\log\sigma(\mathbf{z}_i)_{t_i}
    """
    if dim < 0:
        dim += logits.dims
    gathered = -log_softmax(logits, dim)._gather_index(target, dim)
    return gathered.contiguous().view(*target.shape)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def _tile(image: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape a ``(B, C, H, W)`` image into ``(B, C, Nh, Nw, kh*kw)`` so that a
    pooling window sits on the trailing dimension.

    Returns the tiled tensor plus the pooled ``(Nh, Nw)``.
    """
    batch, channel, height, width = image.shape
    kh, kw = kernel
    if height % kh != 0 or width % kw != 0:
        raise ValueError(
            f"Image HxW {image.shape[-2:]} not divisible by pooling kernel {kernel}"
        )
    nh, nw = height // kh, width // kw
    x = image.contiguous().view(batch, channel, nh, kh, nw, kw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(batch, channel, nh, nw, kh * kw), nh, nw


def avg_pool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling of a ``(B, C, H, W)`` input using a ``kernel`` window."""
    batch, channel, _, _ = input.shape
    x, nh, nw = _tile(input, kernel)
    return x.mean(dim=4).view(batch, channel, nh, nw)


def max_pool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pooling of a ``(B, C, H, W)`` input using a ``kernel`` window."""
    batch, channel, _, _ = input.shape
    x, nh, nw = _tile(input, kernel)
    return x.max(dim=4).view(batch, channel, nh, nw)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def layer_norm(input: Tensor, normalized_shape: Sequence[int], eps: float = 1e-5) -> Tensor:
    """
    Layer normalisation over the last ``len(normalized_shape)`` dimensions.

    Stats are computed per-example (no running buffers are needed), and the
    result is ``(x - mean) / sqrt(var + eps)`` broadcast back to ``x``.
    """
    dims = len(normalized_shape)
    keep_dims = list(range(input.dims - dims, input.dims))

    mean = input
    var = input
    for d in keep_dims:
        mean = mean.mean(dim=d)  # keepdims: reduced dim stays size 1
        var = var.var(dim=d)
    eps_t = input._ensure_tensor(eps)
    return (input - mean) / (var + eps_t).sqrt()


# ---------------------------------------------------------------------------
# Reductions and misc helpers
# ---------------------------------------------------------------------------
def logsumexp(input: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    r"""
    Numerically stable log-sum-exp over ``dim``:

    .. math:: \mathrm{lse}(x) = \max_j x_j + \log\sum_j e^{x_j - \max_j x_j}

    Mirrors the reference behaviour: like the other Torchlight reductions the
    reduced dimension is *kept* at size 1 by default (``keepdim=True``).
    """
    if dim < 0:
        dim += input.dims
    mx = input.max(dim)
    out = (input - mx).exp().sum(dim).log() + mx
    if keepdim:
        return out
    shape = [s for i, s in enumerate(input.shape) if i != dim]
    return out.contiguous().view(*shape)


def argmax(input: Tensor, dim: int = -1) -> Tensor:
    r"""
    One-hot argmax over ``dim``: returns a float tensor of the same shape as
    ``input`` with a ``1.0`` at every position holding the maximum (and
    ``0.0`` elsewhere).  Computed as :math:`\mathbb{1}[x == \max]`, so ties
    all fire (matching the :class:`~torchlight.autograd.functions.Max`
    gradient mask).
    """
    if dim < 0:
        dim += input.dims
    return input.max(dim) == input


def one_hot(input: Tensor, num_classes: int) -> Tensor:
    r"""
    Encode integer indices as one-hot vectors.

    ``input`` of shape ``(*)`` becomes a zero / one tensor of shape
    ``(*, num_classes)`` with a ``1`` on the last axis matching each index.
    The result is a constant (non-differentiable, like its inputs).
    """
    if num_classes < 1:
        raise ValueError(f"one_hot expects num_classes >= 1, got {num_classes}")
    idx = input.to_numpy().astype(np.int64).reshape(-1)
    if idx.size:
        arr = np.eye(num_classes, dtype=np.float32)[idx]
    else:
        arr = np.zeros((0, num_classes), dtype=np.float32)
    return from_numpy(arr.reshape(*input.shape, num_classes), backend=input.backend)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------
def _to_int_pair(x: Union[int, Sequence[int]], name: str) -> Tuple[int, int]:
    """Normalise a scalar / 2-sequence into a ``(first, second)`` pair."""
    if isinstance(x, int):
        return x, x
    if len(x) != 2:
        raise ValueError(f"{name} must be an int or a 2-tuple, got {x!r}")
    return int(x[0]), int(x[1])


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    r"""
    Cross-correlation of a ``(N, C_in, L)`` input with a ``(C_out, C_in, K)``
    weight (plus optional ``(C_out,)`` bias) -> ``(N, C_out, L_out)``.

    ``stride`` / ``padding`` may each be an ``int`` (applied to the sole
    spatial axis).  Backprop flows to both ``input`` and ``weight`` (and
    optionally ``bias``).
    """
    if input.dims != 3:
        raise ValueError(f"conv1d expects a 3D input (N, C, L), got {input.shape}")
    if weight.dims != 3:
        raise ValueError(f"conv1d expects a 3D weight (C_out, C_in, K), got {weight.shape}")
    if input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"conv1d channel mismatch: input has {input.shape[1]} channels, "
            f"weight expects {weight.shape[1]}"
        )
    if not isinstance(stride, int) or not isinstance(padding, int):
        raise TypeError("conv1d stride and padding must be ints")

    out = Conv1d.apply(input, weight, input._ensure_tensor(stride), input._ensure_tensor(padding))
    if bias is not None:
        out = out + bias.contiguous().view(1, weight.shape[0], 1)
    return out


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    r"""
    Cross-correlation of a ``(N, C_in, H, W)`` input with a
    ``(C_out, C_in, KH, KW)`` weight (plus optional ``(C_out,)`` bias)
    -> ``(N, C_out, H_out, W_out)``.

    ``stride`` / ``padding`` may each be an ``int`` or a ``(h, w)`` pair.
    Backprop flows to ``input``, ``weight`` and (optionally) ``bias``.
    """
    if input.dims != 4:
        raise ValueError(f"conv2d expects a 4D input (N, C, H, W), got {input.shape}")
    if weight.dims != 4:
        raise ValueError(
            f"conv2d expects a 4D weight (C_out, C_in, KH, KW), got {weight.shape}"
        )
    if input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"conv2d channel mismatch: input has {input.shape[1]} channels, "
            f"weight expects {weight.shape[1]}"
        )
    sh, sw = _to_int_pair(stride, "stride")
    ph, pw = _to_int_pair(padding, "padding")

    stride_t = input._ensure_tensor([sh, sw])
    padding_t = input._ensure_tensor([ph, pw])
    out = Conv2d.apply(input, weight, stride_t, padding_t)
    if bias is not None:
        out = out + bias.contiguous().view(1, weight.shape[0], 1, 1)
    return out


# ---------------------------------------------------------------------------
# Embedding / variable-length sequence pooling
# ---------------------------------------------------------------------------
def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None) -> Tensor:
    r"""
    Look up embedding rows for each index in ``input``.

    ``weight`` has shape ``(num_embeddings, embedding_dim)``; ``input`` has
    shape ``(*)`` of indices and the result has shape ``(*, embedding_dim)``
    (the indices flatten to select rows, then reshape back).  Differentiable
    with respect to ``weight``: the gradient at row ``i`` is the sum of the
    output gradients of every position that referenced ``i``.

    With ``padding_idx`` set, rows whose index equals it are forced to zero in
    the output (and their gradient is masked out), like ``torch.nn.Embedding``.
    """
    ids = input.contiguous().flatten()
    out = IndexSelect.apply(weight, ids, weight._ensure_tensor(0))
    out = out.reshape(*input.shape, weight.shape[-1])
    if padding_idx is not None:
        pad_mask = (input != padding_idx).contiguous().reshape(*input.shape, 1)
        out = out * pad_mask
    return out


def _expand_mask(mask: Tensor, like: Tensor) -> Tensor:
    """Append size-1 axes so ``mask`` broadcasts as the *leading* dims of ``like``.

    E.g. a ``(B, L)`` mask against a ``(B, L, E)`` input becomes ``(B, L, 1)``
    (this is the alignment numpy needs: trailing axes are the ones broadcast).
    """
    if mask.dims >= like.dims:
        return mask
    extra = like.dims - mask.dims
    return mask.reshape(*mask.shape, *((1,) * extra))


def _drop_axis(shape: Tuple[int, ...], dim: int) -> Tuple[int, ...]:
    """``shape`` with ``dim`` removed (negative ``dim`` resolved)."""
    dim %= len(shape)
    return shape[:dim] + shape[dim + 1 :]


def masked_sum(input: Tensor, mask: Tensor, dim: int) -> Tensor:
    r"""
    Sum ``input`` along ``dim`` keeping only positions where ``mask`` is 1.

    ``input`` has shape ``(*, L, ...)``; ``mask`` has the *leading* ``mask``
    dims of ``input`` (e.g. ``(B, L)`` over a ``(B, L, E)`` input) and is
    broadcast to it as a constant.  Fully masked sums come out as ``0``.
    """
    m = _expand_mask(mask, input)
    reduced = (input * m).sum(dim)
    return reduced.reshape(*_drop_axis(input.shape, dim))


def masked_mean(input: Tensor, mask: Tensor, dim: int) -> Tensor:
    r"""
    Mean over the unmasked positions along ``dim`` (``masked_sum`` / count).

    Rows without any unmasked position yield ``0`` instead of NaN.  ``mask``
    is broadcast exactly as in :func:`masked_sum`.
    """
    m = _expand_mask(mask, input)
    count = m.sum(dim)
    # Guard against division by zero: an all-masked row counts as 1 position.
    count = count + (count == 0.0)
    reduced = (input * m).sum(dim) / count
    return reduced.reshape(*_drop_axis(input.shape, dim))


def masked_max(input: Tensor, mask: Tensor, dim: int) -> Tensor:
    r"""
    Max over the unmasked positions along ``dim``.

    Masked positions are parked at ``-1e9`` before the reduction (they only
    win if every competing value is even smaller).  ``mask`` is broadcast
    exactly as in :func:`masked_sum`.
    """
    m = _expand_mask(mask, input)
    parked = input + (m - 1.0) * 1e9
    reduced = parked.max(dim)
    return reduced.reshape(*_drop_axis(input.shape, dim))