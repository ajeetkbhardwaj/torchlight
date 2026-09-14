"""
Differentiable operations (``Function`` subclasses).

Every operation in Torchlight that must support backpropagation is written
as a :class:`Function` subclass exposing two classmethods::

    forward(ctx, *inputs) -> output_tensor
    backward(ctx, grad_output) -> tuple of grads, one per forward input

:meth:`Function.apply` is the only public entry point; it:

1. detects whether any input requires grad,
2. builds a :class:`Context`,
3. runs ``forward`` on *detached* inputs,
4. records a :class:`History` if gradients are needed, and
5. fires the JIT trace hook.

Backward gradients are *un-reduced*: they may not yet match an input's shape
(e.g. because forward broadcast).  The tensor layer fixes this with
``Tensor._expand_grad`` before accumulating.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Type

import numpy as np

from ..core import ops as scalar_ops
from ..core.tensor_data import IndexingError, TensorData, shape_broadcast
from .autodiff import Context, History, _maybe_record


def _wrap_tuple(x: Any) -> Tuple[Any, ...]:
    """Turn a scalar / single object into a tuple (so backward needn't)."""
    if isinstance(x, tuple):
        return x
    return (x,)


def _as_int(tensor: Any) -> int:
    """Read a 1-element tensor's single value as a python int."""
    return int(tensor.item())


# ---------------------------------------------------------------------------
# Function base
# ---------------------------------------------------------------------------
class Function:
    """Base class for all differentiable operations."""

    @classmethod
    def _forward(cls, ctx: Context, *inps: Any) -> Any:
        return cls.forward(ctx, *inps)  # type: ignore[attr-defined]

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Any) -> Tuple[Any, ...]:
        return _wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore[attr-defined]

    @classmethod
    def apply(cls, *vals: Any) -> Any:
        """
        Run the forward pass through this function, wiring autograd.

        Args:
            *vals: the input tensors (and constant metadata tensors).

        Returns:
            A tensor equal to the output of ``forward``.  If any input
            requires gradient, the returned tensor carries a :class:`History`
            so backprop can reach the leaves.
        """
        need_grad = any(getattr(v, "requires_grad", False) for v in vals)
        raw_vals = tuple(v.detach() for v in vals)

        ctx = Context(no_grad=not need_grad)
        out = cls._forward(ctx, *raw_vals)

        if need_grad:
            # Rebuild the output carrying graph metadata.  Parents are the
            # *original* inputs so traversal reaches the leaves.
            from ..tensor.tensor import Tensor  # deferred: breaks circularity

            out = Tensor(
                out._tensor,
                history=History(last_fn=cls, ctx=ctx, inputs=tuple(vals)),
                backend=out.backend,
            )

        _maybe_record(cls, tuple(vals), out)
        return out


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------
class Neg(Function):
    r"""Negation :math:`f(x) = -x`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        return t1._new(t1.backend.neg_map(t1._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        return grad_output._new(grad_output.backend.neg_map(grad_output._tensor))


class Inv(Function):
    r"""Reciprocal :math:`f(x) = 1/x`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        ctx.save_for_backward(t1)
        return t1._new(t1.backend.inv_map(t1._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.inv_back(t1._tensor, grad_output._tensor))


class Exp(Function):
    r"""Exponential :math:`f(x) = e^x`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        out = t1._new(t1.backend.exp_map(t1._tensor))
        ctx.save_for_backward(out)  # dz/dx = e^x = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (out,) = ctx.saved_tensors
        return out._new(out.backend.mul(out._tensor, grad_output._tensor))


class Log(Function):
    r"""Natural log :math:`f(x) = \log(x + 10^{-6})` (shifted for stability)."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        ctx.save_for_backward(t1)
        return t1._new(t1.backend.log_map(t1._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.log_back(t1._tensor, grad_output._tensor))


class Sigmoid(Function):
    r"""Sigmoid :math:`\sigma(x) = 1/(1 + e^{-x})` (stable)."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        out = t1._new(t1.backend.sigmoid_map(t1._tensor))
        ctx.save_for_backward(out)  # dz/dx = sigma(1 - sigma)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (out,) = ctx.saved_tensors
        return out._new(out.backend.sigmoid_back(out._tensor, grad_output._tensor))


class Tanh(Function):
    r"""Hyperbolic tangent :math:`f(x) = \tanh(x)`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        out = t1._new(t1.backend.tanh_map(t1._tensor))
        ctx.save_for_backward(out)  # dz/dx = 1 - tanh^2
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (out,) = ctx.saved_tensors
        # (1 - out^2) * grad
        one = out._new(TensorData.ones(out._tensor.shape, out._tensor.dtype))
        sq = out._new(out.backend.mul(out._tensor, out._tensor))
        coeff = out._new(out.backend.sub(one._tensor, sq._tensor))
        return out._new(out.backend.mul(coeff._tensor, grad_output._tensor))


class ReLU(Function):
    r"""Rectified linear unit :math:`f(x) = \max(x, 0)`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        ctx.save_for_backward(t1)
        return t1._new(t1.backend.relu_map(t1._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.relu_back(t1._tensor, grad_output._tensor))


class LeakyReLU(Function):
    r"""Leaky ReLU with slope :math:`\alpha`: :math:`f(x) = x` if
    :math:`x>0` else :math:`\alpha x`."""

    alpha: float = 0.01

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        out = t1._new(t1.backend.map(scalar_ops.leaky_relu)(t1._tensor))
        ctx.save_for_backward(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.zip(scalar_ops.leaky_relu_back)(t1._tensor, grad_output._tensor))


class Sqrt(Function):
    r"""Square root :math:`f(x) = \sqrt{x}`, gradient :math:`1/(2\sqrt{x})`
    (guarded against :math:`x=0`)."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        out = t1._new(t1.backend.sqrt_map(t1._tensor))
        ctx.save_for_backward(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.sqrt_back(t1._tensor, grad_output._tensor))


class Abs(Function):
    r"""Absolute value :math:`f(x) = |x|`, gradient :math:`\mathrm{sign}(x)`."""

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        ctx.save_for_backward(t1)
        return t1._new(t1.backend.abs_map(t1._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        (t1,) = ctx.saved_tensors
        return t1._new(t1.backend.abs_back(t1._tensor, grad_output._tensor))


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------
class Add(Function):
    r"""Elementwise addition :math:`f(a, b) = a + b` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        return t1._new(t1.backend.add(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return grad_output, grad_output


class Mul(Function):
    r"""Elementwise multiplication :math:`f(a, b) = a \cdot b` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        ctx.save_for_backward(t1, t2)
        return t1._new(t1.backend.mul(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        t1, t2 = ctx.saved_tensors
        return (
            t1._new(t1.backend.mul(t2._tensor, grad_output._tensor)),
            t2._new(t2.backend.mul(t1._tensor, grad_output._tensor)),
        )


class PowerScalar(Function):
    r"""
    Power :math:`f(a, b) = a^b`.

    The exponent ``b`` may be a scalar tensor ``(1,)`` -- or any tensor; the
    elementwise numpy ``power`` broadcasts it.  Used for things like
    ``x ** 2`` and Adam's ``grad ** 2``.  Both gradients are provided
    (:math:`\partial a^b / \partial a = b\,a^{b-1}`, and the exponent part
    requires :math:`a > 0`).
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        ctx.save_for_backward(t1, t2)
        return t1._new(t1.backend.pow(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        t1, t2 = ctx.saved_tensors
        b = t2  # exponent tensor

        a_pow = t1._new(t1.backend.pow(t1._tensor, b._tensor))
        # d/da a^b = b * a^(b-1)   (computed as a^(b-1), never a^b / a, so a=0
        # stays well-defined for integer-ish exponents)
        one = t1._new(TensorData.ones(b._tensor.shape, b._tensor.dtype))
        b_minus_1 = t1._new(t1.backend.sub(b._tensor, one._tensor))
        a_pow_m1 = t1._new(t1.backend.pow(t1._tensor, b_minus_1._tensor))
        grad_a = t1._new(
            t1.backend.mul(
                t1._new(t1.backend.mul(b._tensor, a_pow_m1._tensor))._tensor,
                grad_output._tensor,
            )
        )

        # d/db a^b = a^b * log(a)   (valid for a > 0)
        if b._tensor.size == 1:
            grad_b = 0.0
        else:
            log_a = t1._new(t1.backend.map(np.log)(t1._tensor))
            grad_b = t1._new(t1.backend.mul(t1._new(t1.backend.mul(a_pow._tensor, log_a._tensor))._tensor,
                                            grad_output._tensor))
        return grad_a, grad_b


class MatMul(Function):
    r"""
    (Batched) matrix product :math:`f(A, B) = A \cdot B`.

    Backprop uses the classic identities:

    .. math::
        \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^\top
        \qquad
        \frac{\partial L}{\partial B} = A^\top \cdot \frac{\partial L}{\partial C}

    with the last two dimensions swapped on the non-gradient side.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        ctx.save_for_backward(t1, t2)
        return t1._new(t1.backend.matmul(t1._tensor, t2._tensor))

    @staticmethod
    def _transpose(t: Any) -> Any:
        order = list(range(t.dims))
        order[-2], order[-1] = order[-1], order[-2]
        return t._new(t._tensor.permute(*order))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        t1, t2 = ctx.saved_tensors
        go = grad_output
        t2_t = MatMul._transpose(t2)
        t1_t = MatMul._transpose(t1)
        return (
            t1._new(t1.backend.matmul(go._tensor, t2_t._tensor)),
            t2._new(t2.backend.matmul(t1_t._tensor, go._tensor)),
        )


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------
class Sum(Function):
    """
    Sum reduction along one dimension.

    The reduced dimension is *kept* at size 1 (keepdims), matching how the
    rest of the framework broadcasts.  ``dim`` is passed as a 1-element
    constant tensor.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, dim: Any) -> Any:
        ctx.save_for_backward(t1.shape)
        return t1._new(t1.backend.add_reduce(t1._tensor, _as_int(dim)))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        (orig_shape,) = ctx.saved_tensors
        # ``_expand_grad`` in the tensor layer broadcasts this back to
        # ``orig_shape``, distributing the gradient uniformly over the
        # reduced dimension (which is exactly d sum/dx = 1).
        return grad_output, 0.0


class All(Function):
    """
    Product-reduction along one dimension -- used for ``all`` (boolean
    conjunction converted to floats).  Non-differentiable.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, dim: Any) -> Any:
        return t1._new(t1.backend.mul_reduce(t1._tensor, _as_int(dim)))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return 0.0, 0.0


class Max(Function):
    r"""
    Maximum reduction along one dimension (keepdims).

    Backward routes the gradient only to the argmax position(s):
    :math:`\nabla = \mathbb{1}[x == \max] \cdot g`.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, dim: Any) -> Any:
        out = t1._new(t1.backend.max_reduce(t1._tensor, _as_int(dim)))
        ctx.save_for_backward(t1, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        t1, out = ctx.saved_tensors
        mask = t1._new(t1.backend.eq(t1._tensor, out._tensor))  # broadcasts
        return t1._new(t1.backend.mul(mask._tensor, grad_output._tensor)), 0.0


class Min(Function):
    """
    Minimum reduction along one dimension (keepdims).

    Backward routes the gradient only to the argmin position(s):
    :math:`\\nabla = \\mathbb{1}[x == \\min] \\cdot g`.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, dim: Any) -> Any:
        neg = t1._new(t1.backend.neg_map(t1._tensor))
        mx = neg._new(neg.backend.max_reduce(neg._tensor, _as_int(dim)))
        out = mx._new(mx.backend.neg_map(mx._tensor))
        ctx.save_for_backward(t1, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        t1, out = ctx.saved_tensors
        mask = t1._new(t1.backend.eq(t1._tensor, out._tensor))  # broadcasts
        return t1._new(t1.backend.mul(mask._tensor, grad_output._tensor)), 0.0


# ---------------------------------------------------------------------------
# Layout ops
# ---------------------------------------------------------------------------
class View(Function):
    """
    Reshape a (contiguous) tensor to a new shape with the same number of
    elements.  Shape is supplied as a constant 1-D tensor of ints.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, shape: Any) -> Any:
        ctx.save_for_backward(t1.shape)
        if not t1._tensor.is_contiguous():
            raise IndexingError("Must be contiguous to view; call .contiguous() first")
        new_shape = tuple(int(shape[i]) for i in range(shape.size))
        if tensor_prod(t1._tensor.shape) != tensor_prod(new_shape):
            raise IndexingError(f"Cannot view {t1.shape} as {new_shape}")
        return t1._new(t1._tensor.reshape(new_shape))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        (orig_shape,) = ctx.saved_tensors
        return grad_output._new(grad_output._tensor.reshape(orig_shape)), 0.0


class Permute(Function):
    """
    Reorder tensor dimensions.  Order supplied as a constant 1-D tensor.
    Backward applies the inverse permutation.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any, order: Any) -> Any:
        ctx.save_for_backward([int(order[i]) for i in range(order.size)])
        return t1._new(t1._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        (order,) = ctx.saved_tensors
        inverse = [0] * len(order)
        for i, o in enumerate(order):
            inverse[o] = i
        return grad_output._new(grad_output._tensor.permute(*inverse)), 0.0


class Contiguous(Function):
    """
    Materialise a dense (contiguous) copy of a tensor.  The identity op for
    data, but it also *fixes* stride-0 / permuted layouts.
    """

    @staticmethod
    def forward(ctx: Context, t1: Any) -> Any:
        return t1._new(t1._tensor.contiguous())

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        return grad_output


# ---------------------------------------------------------------------------
# Comparison ops (non-differentiable: they return zero gradients)
# ---------------------------------------------------------------------------
class LT(Function):
    r"""Elementwise :math:`\mathbb{1}[a < b]` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        return t1._new(t1.backend.lt(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return 0.0, 0.0


class GT(Function):
    r"""Elementwise :math:`\mathbb{1}[a > b]` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        return t1._new(t1.backend.gt(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return 0.0, 0.0


class EQ(Function):
    r"""Elementwise :math:`\mathbb{1}[a == b]` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        return t1._new(t1.backend.eq(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return 0.0, 0.0


class IsClose(Function):
    r"""Elementwise :math:`\mathbb{1}[|a - b| < 10^{-2}]` (broadcasts)."""

    @staticmethod
    def forward(ctx: Context, t1: Any, t2: Any) -> Any:
        return t1._new(t1.backend.is_close(t1._tensor, t2._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Gather (used by cross-entropy with index targets)
# ---------------------------------------------------------------------------
class Gather(Function):
    """
    Select elements along a dim by an index tensor.

    ``forward`` is ``np.take_along_axis`` and ``backward`` scatters the
    gradient back with ``np.put_along_axis`` (a one-hot style scatter), so
    it behaves exactly like PyTorch's ``torch.gather``.
    """

    @staticmethod
    def forward(ctx: Context, src: Any, index: Any, dim: Any) -> Any:
        dim_i = _as_int(dim)
        idx = index._tensor.to_numpy().astype(np.int64)
        ctx.save_for_backward(src, idx, dim_i)
        arr = src._tensor.to_numpy()
        out = np.take_along_axis(arr, idx, axis=dim_i)
        return src._new(TensorData(np.ascontiguousarray(out), tuple(out.shape)))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any, Any]:
        src, idx, dim_i = ctx.saved_tensors
        grad = np.zeros(src.shape, dtype=np.float32)
        np.put_along_axis(grad, idx, grad_output._tensor.to_numpy(), axis=dim_i)
        return (
            src._new(TensorData(grad, tuple(src.shape))),
            0.0,
            0.0,
        )


# ---------------------------------------------------------------------------
# Fused ops (attention softmax / layer norm)
# ---------------------------------------------------------------------------
class AttnSoftmax(Function):
    r"""
    Row-wise softmax over the last axis of ``x + mask``.

    Mirrors the fused ``ker_attn_softmax`` kernels: the max is subtracted for
    numerical stability and a tiny ``1e-8`` is added to the normalising sum,
    so the op is a drop-in replacement for the ``MinTorch`` fused attention
    softmax.  ``mask`` is constant (like adding an additive attention mask) and
    therefore non-differentiable.
    """

    @staticmethod
    def forward(ctx: Context, a: Any, mask: Any) -> Any:
        out = a._new(a.backend.attn_softmax_fw(a._tensor, mask._tensor))
        ctx.save_for_backward(out)  # softmax output reused by backward
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any]:
        (soft,) = ctx.saved_tensors
        d = grad_output._new(soft.backend.attn_softmax_bw(grad_output._tensor, soft._tensor))
        return d, 0.0


class LayerNorm(Function):
    r"""
    Layer normalisation over the last axis:

    .. math::
        \hat{x} = \frac{x - \mathrm{mean}(x)}{\sqrt{\mathrm{var}(x) + \epsilon}}
        ,\qquad
        y = \hat{x} \odot \gamma + \beta

    ``gamma`` and ``beta`` are per-hidden-dim parameters (shape = the last
    axis), all reduction sums in backward accumulate over the leading axes --
    exactly the fused LightSeq ``ker_ln_bw_*`` semantics.
    """

    @staticmethod
    def forward(ctx: Context, a: Any, gamma: Any, beta: Any) -> Any:
        ctx.save_for_backward(a, gamma)
        return a._new(a.backend.layernorm_fw(a._tensor, gamma._tensor, beta._tensor))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any, Any]:
        a, gamma = ctx.saved_tensors
        dinp, dgamma, dbeta = a.backend.layernorm_bw(
            grad_output._tensor, a._tensor, gamma._tensor, gamma._tensor
        )
        return (
            a._new(dinp),
            gamma._new(dgamma),
            gamma._new(dbeta),
        )


def tensor_prod(shape: Any) -> int:
    """Product of a shape tuple."""
    out = 1
    for s in shape:
        out *= s
    return out


__all__ = [
    "Function",
    "Neg",
    "Inv",
    "Exp",
    "Log",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LeakyReLU",
    "Sqrt",
    "Abs",
    "Add",
    "Mul",
    "PowerScalar",
    "MatMul",
    "Sum",
    "All",
    "Max",
    "Min",
    "View",
    "Permute",
    "Contiguous",
    "LT",
    "GT",
    "EQ",
    "IsClose",
    "Gather",
    "AttnSoftmax",
    "LayerNorm",
]