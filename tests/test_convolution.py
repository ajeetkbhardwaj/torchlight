"""Convolution tests: forward correctness, gradient checks, and modules."""

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import functional as F
from torchlight.nn import Conv1d, Conv2d

from helpers import assert_allclose_grad


def ref_conv1d(x, w, stride=1, padding=0):
    """Naive (loop) convolution reference."""
    n, c, l = x.shape
    co, ci, k = w.shape
    lp = l + 2 * padding
    lo = (lp - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((n, co, lo))
    for o in range(co):
        for ll in range(lo):
            for cc in range(ci):
                for kk in range(k):
                    out[:, o, ll] += xp[:, cc, ll * stride + kk] * w[o, cc, kk]
    return out


def ref_conv2d(x, w, sh=1, sw=1, ph=0, pw=0):
    """Naive (loop) 2D convolution reference."""
    n, c, h, wd = x.shape
    co, ci, kh, kw = w.shape
    hp, wp = h + 2 * ph, wd + 2 * pw
    ho, wo = (hp - kh) // sh + 1, (wp - kw) // sw + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((n, co, ho, wo))
    for o in range(co):
        for hh in range(ho):
            for ww in range(wo):
                for cc in range(ci):
                    for khh in range(kh):
                        for kww in range(kw):
                            out[:, o, hh, ww] += (
                                xp[:, cc, hh * sh + khh, ww * sw + kww] * w[o, cc, khh, kww]
                            )
    return out


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 0), (2, 1), (3, 2)])
def test_conv1d_matches_reference(stride, padding):
    rng = np.random.default_rng(0)
    x = tl.tensor(rng.random((2, 3, 9)).astype(np.float32))
    w = tl.tensor(rng.random((4, 3, 3)).astype(np.float32))
    out = F.conv1d(x, w, stride=stride, padding=padding)
    ref = ref_conv1d(x.to_numpy(), w.to_numpy(), stride, padding)
    np.testing.assert_allclose(out.to_numpy(), ref, atol=1e-5)


@pytest.mark.parametrize(
    "stride,padding",
    [((1, 1), (0, 0)), ((1, 1), (1, 0)), ((2, 1), (1, 0)), ((2, 2), (1, 1))],
)
def test_conv2d_matches_reference(stride, padding):
    rng = np.random.default_rng(1)
    x = tl.tensor(rng.random((2, 2, 6, 7)).astype(np.float32))
    w = tl.tensor(rng.random((3, 2, 3, 2)).astype(np.float32))
    out = F.conv2d(x, w, stride=stride, padding=padding)
    ref = ref_conv2d(x.to_numpy(), w.to_numpy(), stride[0], stride[1], padding[0], padding[1])
    np.testing.assert_allclose(out.to_numpy(), ref, atol=1e-5)


def test_conv1d_bias():
    rng = np.random.default_rng(2)
    x = tl.tensor(rng.random((2, 3, 8)).astype(np.float32))
    w = tl.tensor(rng.random((4, 3, 3)).astype(np.float32))
    b = tl.tensor(rng.random((4,)).astype(np.float32))
    out = F.conv1d(x, w, b)
    ref = ref_conv1d(x.to_numpy(), w.to_numpy()) + b.to_numpy()[:, None]
    np.testing.assert_allclose(out.to_numpy(), ref, atol=1e-5)
    # bias gradients flow back through the addition: each bias collects the
    # mean's 1/(N*C_out*L_out) contribution over its N*L_out outputs (2*6 of 48)
    b.requires_grad_(True)
    F.conv1d(x, w, b).mean().backward()
    assert b.grad is not None
    np.testing.assert_allclose(b.grad.to_numpy(), np.full(4, 12.0 / 48.0), atol=1e-6)


def test_conv1d_grad_input():
    rng = np.random.default_rng(3)
    x0 = rng.random((1, 2, 6)).astype(np.float32)
    w = tl.tensor(rng.random((2, 2, 3)).astype(np.float32))
    xg = tl.tensor(x0)
    xg.requires_grad_(True)
    F.conv1d(xg, w).sum().backward()
    assert_allclose_grad(xg, x0, lambda t: F.conv1d(t, w).sum())
    xg.zero_grad_()
    F.conv1d(xg, w, stride=2, padding=1).sum().backward()
    assert_allclose_grad(xg, x0, lambda t: F.conv1d(t, w, stride=2, padding=1).sum())


def test_conv1d_grad_weight():
    rng = np.random.default_rng(4)
    x = tl.tensor(rng.random((2, 2, 7)).astype(np.float32))
    w0 = rng.random((3, 2, 3)).astype(np.float32)
    wg = tl.tensor(w0)
    wg.requires_grad_(True)
    F.conv1d(x, wg, stride=2, padding=1).sum().backward()
    assert_allclose_grad(wg, w0, lambda t: F.conv1d(x, t, stride=2, padding=1).sum())


def test_conv2d_grad_input():
    rng = np.random.default_rng(5)
    x0 = rng.random((1, 2, 5, 6)).astype(np.float32)
    w = tl.tensor(rng.random((2, 2, 2, 2)).astype(np.float32))
    xg = tl.tensor(x0)
    xg.requires_grad_(True)
    F.conv2d(xg, w, stride=2, padding=1).sum().backward()
    assert_allclose_grad(xg, x0, lambda t: F.conv2d(t, w, stride=2, padding=1).sum())


def test_conv2d_grad_weight():
    rng = np.random.default_rng(6)
    x = tl.tensor(rng.random((2, 2, 6, 6)).astype(np.float32))
    w0 = rng.random((3, 2, 2, 3)).astype(np.float32)
    wg = tl.tensor(w0)
    wg.requires_grad_(True)
    F.conv2d(x, wg).sum().backward()
    assert_allclose_grad(wg, w0, lambda t: F.conv2d(x, t).sum())


def test_conv1d_module_forward_and_parameters():
    layer = Conv1d(3, 4, 3, stride=1, padding=1)
    assert layer.weight.shape == (4, 3, 3)
    assert layer.bias.shape == (4,)
    out = layer(tl.zeros((2, 3, 10)))
    assert out.shape == (2, 4, 10)
    # parameters are leaves that backprop into
    out.sum().backward()
    assert layer.weight.value.grad is not None
    assert layer.bias.value.grad is not None


def test_conv2d_module_forward_and_parameters():
    layer = Conv2d(2, 4, (3, 2), stride=(2, 1), padding=(1, 0))
    assert layer.weight.shape == (4, 2, 3, 2)
    assert layer.bias.shape == (4,)
    out = layer(tl.zeros((2, 2, 8, 9)))
    assert out.shape == (2, 4, 4, 8)
    out.sum().backward()
    assert layer.weight.value.grad is not None
    assert layer.bias.value.grad is not None


def test_conv_module_no_bias():
    c1 = Conv1d(3, 4, 3, bias=False)
    assert c1.bias is None
    assert len(c1.parameters()) == 1
    c2 = Conv2d(2, 4, 3, bias=False)
    assert c2.bias is None
    assert len(c2.parameters()) == 1


def test_conv_rejects_bad_shapes():
    x = tl.zeros((2, 3, 16))
    w = tl.zeros((4, 5, 3))
    with pytest.raises(ValueError):
        F.conv1d(x, w)
    with pytest.raises(ValueError):
        F.conv1d(tl.zeros((2, 3)), w)


def test_conv_module_trains():
    """A single Conv1d layer can fit a linear mapping (sanity), end-to-end."""
    rng = np.random.default_rng(7)
    layer = Conv1d(1, 1, 3, padding=1)  # K=3, padding=1 keeps length
    x_np = rng.random((8, 1, 10)).astype(np.float32)
    target = 2.0 * layer(tl.tensor(x_np)).detach()
    opt = tl.optim.SGD(layer.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss = ((layer(tl.tensor(x_np)) - target) ** 2).mean()
        loss.backward()
        opt.step()
    final = (layer(tl.tensor(x_np)) - target).abs().mean().item()
    assert final < 1e-2