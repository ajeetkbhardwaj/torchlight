"""data suite: datasets, dataloaders, and the synthetic-problem generators."""

import numpy as np
import pytest

from torchlight.data import DataLoader, TensorDataset, make_synthetic
from torchlight.data import synthetic as synth
from torchlight import tensor


def test_tensor_dataset_indexing_shapes():
    X = tensor(np.arange(10.0).reshape(5, 2))
    y = tensor(np.arange(5.0))
    ds = TensorDataset(X, y)
    assert len(ds) == 5
    x0, y0 = ds[0]
    assert x0.shape == (2,)
    assert float(y0.to_numpy().reshape(-1)[0]) == 0.0


def test_tensor_dataset_requires_same_length():
    with pytest.raises(ValueError):
        TensorDataset(tensor(np.zeros((5, 2))), tensor(np.zeros((3,))))


def test_dataloader_batches_and_shapes():
    X = tensor(np.arange(20.0).reshape(10, 2))
    y = tensor(np.arange(10.0))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True, seed=0)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (4, 2)
    assert batches[0][1].shape == (4, 1)
    assert batches[-1][0].shape == (2, 2)


def test_dataloader_drop_last():
    X = tensor(np.arange(20.0).reshape(10, 2))
    loader = DataLoader(TensorDataset(X, X), batch_size=4, drop_last=True)
    assert len(loader) == 2
    assert len(list(loader)) == 2


def test_dataloader_shuffle_changes_order_with_fixed_seed():
    X = tensor(np.arange(20.0).reshape(10, 2))
    y = tensor(np.arange(10.0))
    ds = TensorDataset(X, y)
    a = [b[0].to_numpy().copy() for b in DataLoader(ds, batch_size=10, shuffle=True, seed=7)]
    b = [b[0].to_numpy().copy() for b in DataLoader(ds, batch_size=10, shuffle=True, seed=7)]
    # (batch of all 10 samples, so comparing first batch across runs)
    assert np.allclose(a[0], b[0])
    assert not np.allclose(a[0], np.arange(20.0).reshape(10, 2)[:10])


@pytest.mark.parametrize("name,n", [
    ("Simple", 50), ("Diag", 50), ("Split", 60), ("Xor", 80), ("Circle", 70), ("Spiral", 100),
])
def test_synthetic_generators_build_graphs(name, n):
    g = synth._DATASETS[name](n)
    assert g.N == n
    assert len(g.X) == n
    assert len(g.y) == n
    assert set(g.y) == {0, 1}
    for x1, x2 in g.X:
        assert 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0


def test_make_synthetic_returns_tensor_dataset():
    ds = make_synthetic("Xor", n=64, seed=1)
    X, y = ds.to_batch()
    assert X.shape == (64, 2)
    assert y.shape == (64,)


def test_unknown_synthetic_raises():
    with pytest.raises(ValueError):
        make_synthetic("Nope")