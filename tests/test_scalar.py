"""Scalar autodiff tests: forward values, backprop, and derivative checks."""

import numpy as np
import pytest

from torchlight.scalar import Scalar, ScalarFunction, derivative_check


def test_scalar_arithmetic_values():
    x = Scalar(4.0)
    y = Scalar(2.0)
    assert x.data == 4.0
    assert (x * y).data == 8.0
    assert (x + y).data == 6.0
    assert (x - y).data == 2.0
    assert (x / y).data == 2.0
    assert (-x).data == -4.0
    assert (x + 1.0).data == 5.0
    assert (1.0 + x).data == 5.0
    assert (x * 3.0).data == 12.0
    assert (10.0 - x).data == 6.0


def test_scalar_functionals_values():
    x = Scalar(2.0)
    assert x.log().data == pytest.approx(np.log(2.0 + 1e-6))
    assert x.exp().data == pytest.approx(np.exp(2.0))
    assert x.sigmoid().data == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert x.tanh().data == pytest.approx(np.tanh(2.0))
    assert Scalar(-3.0).relu().data == 0.0
    assert Scalar(3.0).relu().data == 3.0


def test_scalar_comparisons():
    x = Scalar(3.0)
    y = Scalar(4.0)
    assert (x < y).data == 1.0
    assert (x > y).data == 0.0
    assert (x == Scalar(3.0)).data == 1.0
    assert (x == y).data == 0.0
    assert bool(x) is True
    assert bool(Scalar(0.0)) is False


def test_scalar_backprop_mul_and_add():
    x = Scalar(4.0)
    y = Scalar(2.0)
    z = x * y + x  # dz/dx = y + 1 = 3, dz/dy = x = 4
    z.backward()
    assert x.derivative == pytest.approx(3.0)
    assert y.derivative == pytest.approx(4.0)


def test_scalar_backprop_chain():
    x = Scalar(2.0)
    z = x.exp().relu()  # dz/dx = e^{2} (relu active)
    z.backward()
    assert x.derivative == pytest.approx(np.exp(2.0), rel=1e-3)


def test_scalar_sigmoid_gradient():
    x = Scalar(0.5)
    z = x.sigmoid()
    z.backward()
    s = 1.0 / (1.0 + np.exp(-0.5))
    assert x.derivative == pytest.approx(s * (1.0 - s), rel=1e-3)


def test_scalar_gradient_sums_at_reused_leaf():
    x = Scalar(3.0)
    z = x * x  # leaf used twice: dz/dx = 2x
    z.backward()
    assert x.derivative == pytest.approx(6.0)


def test_scalar_derivative_check_basic():
    derivative_check(lambda x, y: x * y + x.sigmoid(), Scalar(0.5), Scalar(2.0))


def test_scalar_derivative_check_exp_log():
    derivative_check(lambda x: x.exp() / (1.0 + x.exp()), Scalar(-0.7))
    derivative_check(lambda x: x.log() * x, Scalar(1.4))


def test_scalar_derivative_check_mixed():
    derivative_check(
        lambda x, y: (x * y).tanh() + y.relu() - x / 2.0,
        Scalar(0.3),
        Scalar(-1.2),
    )


def test_scalar_requires_grad_flags():
    x = Scalar(5.0)
    assert x.requires_grad()
    x.requires_grad_(False)
    assert not x.requires_grad()
    assert x.is_constant()
    x.requires_grad_()
    assert x.requires_grad()
    assert x.is_leaf()


def test_scalar_constants_do_not_error_on_backward():
    # A scalar forged with back=None is a constant: backward just no-ops.
    c = Scalar(3.0, None)
    c.backward()
    assert c.derivative is None
    assert c.is_constant()


def test_scalar_function_apply_record():
    from torchlight.scalar.scalar_functions import Mul, Neg

    assert ScalarFunction.__name__ == "ScalarFunction"

    # apply records history and welds gradients
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = Mul.apply(a, b)
    assert c.data == 6.0
    assert c.is_leaf() is False
    out = (c + Neg.apply(a)).data
    assert out == 4.0

    z = Mul.apply(a, b)
    z.backward()
    assert a.derivative == pytest.approx(3.0)
    assert b.derivative == pytest.approx(2.0)


def test_scalar_constant_via_function_apply():
    """Float constants fed to apply are wrapped (tracked) but harmless."""
    from torchlight.scalar.scalar_functions import Add

    x = Scalar(4.0)
    z = Add.apply(x, 10.0)  # a numeric literal gets wrapped into a leaf
    assert z.data == 14.0
    z.backward()  # must not raise when gradients hit the wrapped constant
    assert x.derivative == pytest.approx(1.0)