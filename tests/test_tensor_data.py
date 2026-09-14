"""Tests for the raw data container: shapes, strides, views, broadcasting."""

import numpy as np
import pytest

from torchlight.core.tensor_data import (
    TensorData,
    shape_broadcast,
    strides_from_shape,
)


def test_contiguous_shapes_and_strides():
    td = TensorData(np.arange(6, dtype=np.float32), (2, 3))
    assert td.shape == (2, 3)
    assert td.strides == strides_from_shape((2, 3))
    assert td.is_contiguous()
    assert td.to_numpy().shape == (2, 3)
    assert np.allclose(td.to_numpy(), np.arange(6).reshape(2, 3))


def test_broadcast_view_is_stride_zero_and_noncontiguous():
    td = TensorData(np.array([1.0, 2.0, 3.0], dtype=np.float32), (3,))
    b = td.broadcast_to((4, 3))
    assert b.shape == (4, 3)
    assert b.strides == (0, 1)
    assert not b.is_contiguous()
    assert np.allclose(b.to_numpy(), np.tile([1.0, 2.0, 3.0], (4, 1)))


def test_permute_view_reorders_without_copy():
    td = TensorData(np.arange(6, dtype=np.float32), (2, 3))
    p = td.permute(1, 0)
    assert p.shape == (3, 2)
    assert p.strides == (1, 3)
    assert not p.is_contiguous()
    assert np.allclose(p.to_numpy(), np.arange(6).reshape(2, 3).T)
    # A round-trip permute restores contiguity.
    assert td.permute(1, 0).permute(1, 0).is_contiguous()


def test_reshape_reuses_storage_when_contiguous():
    td = TensorData(np.arange(12, dtype=np.float32), (3, 4))
    r = td.reshape((4, 3))
    assert r.is_contiguous()
    assert np.allclose(r.to_numpy(), np.arange(12).reshape(4, 3))


def test_reshape_materialises_when_not_contiguous():
    td = TensorData(np.arange(12, dtype=np.float32), (3, 4))
    p = td.permute(1, 0)  # (4, 3) strided view
    r = p.reshape((6, 2))
    assert np.allclose(r.to_numpy(), np.arange(12).reshape(3, 4).T.reshape(6, 2))


def test_indexing_scalar():
    td = TensorData(np.arange(6, dtype=np.float32), (2, 3))
    assert td[1, 2] == pytest.approx(5.0)
    td[0, 1] = 99.0
    assert td.get((0, 1)) == pytest.approx(99.0)
    assert np.allclose(td.to_numpy(), [[0, 99, 2], [3, 4, 5]])


@pytest.mark.parametrize("s1,s2,expect", [
    ((1,), (4,), (4,)),
    ((2, 3), (3,), (2, 3)),
    ((1, 3), (2, 1), (2, 3)),
    ((3, 1, 2), (5, 1), (3, 5, 2)),
])
def test_shape_broadcast(s1, s2, expect):
    assert shape_broadcast(s1, s2) == expect


def test_shape_broadcast_rejects_incompatible():
    with pytest.raises(Exception):
        shape_broadcast((2, 3), (4, 5))


def test_strides_match_numpy_row_major():
    assert strides_from_shape((2, 3, 4)) == (12, 4, 1)