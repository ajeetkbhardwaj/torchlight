"""
Synthetic datasets for quick experiments (a port of MiniTorch's zoo).

Every generator returns a :class:`Graph` -- ``N`` 2-D points ``(x, y)`` in
``[0, 1]^2`` plus binary labels -- which is ideal for testing: these are the
problems the MiniTorch course uses to validate nonlinear classifiers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from ..tensor import Tensor, from_numpy
import numpy as np
from .dataset import TensorDataset


@dataclass
class Graph:
    """A labelled 2-D dataset.

    Attributes:
        N: number of points.
        X: ``N`` coordinate pairs in ``[0, 1]^2``.
        y: ``N`` binary labels (``0``/``1``).
    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Return ``N`` uniform random points in ``[0, 1]^2``."""
    return [(random.random(), random.random()) for _ in range(N)]


def simple(N: int) -> Graph:
    """Label 1 iff :math:`x_1 < 0.5` (linearly separable)."""
    X = make_pts(N)
    return Graph(N, X, [1 if x1 < 0.5 else 0 for x1, _ in X])


def diag(N: int) -> Graph:
    """Label 1 iff :math:`x_1 + x_2 < 0.5` (diagonal split)."""
    X = make_pts(N)
    return Graph(N, X, [1 if x1 + x2 < 0.5 else 0 for x1, x2 in X])


def split(N: int) -> Graph:
    """Label 1 iff :math:`x_1 < 0.2` or :math:`x_1 > 0.8` (two clusters)."""
    X = make_pts(N)
    return Graph(N, X, [1 if x1 < 0.2 or x1 > 0.8 else 0 for x1, _ in X])


def xor(N: int) -> Graph:
    """Label 1 iff exactly one of :math:`x_1, x_2 > 0.5` (XOR)."""
    X = make_pts(N)
    return Graph(
        N, X, [1 if (x1 < 0.5 and x2 > 0.5) or (x1 > 0.5 and x2 < 0.5) else 0 for x1, x2 in X]
    )


def circle(N: int) -> Graph:
    """Label 1 iff the point lies outside a disc of radius ~0.316 around (0.5, 0.5)."""
    X = make_pts(N)
    y = []
    for x1, x2 in X:
        d = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
        y.append(1 if d > 0.1 else 0)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Two interleaved spiral arms (the hardest of the bunch)."""

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    half = N // 2
    arm1 = [(x(10.0 * (float(i) / half)) + 0.5, y(10.0 * (float(i) / half)) + 0.5) for i in range(half)]
    arm2 = [(y(-10.0 * (float(i) / half)) + 0.5, x(-10.0 * (float(i) / half)) + 0.5) for i in range(half)]
    X = arm1 + arm2
    labels = [0] * half + [1] * (N - half)
    return Graph(N, X, labels)


def make_spiral(N: int) -> List[Tuple[float, float]]:
    """Compatibility alias returning just the points of the spiral."""
    return spiral(N).X


#: Registry of available problems, keyed by name (MiniTorch-compatible).
_DATASETS: Dict[str, Callable[[int], Graph]] = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}

_LOOKUP: Dict[str, str] = {k.casefold(): k for k in _DATASETS}


def make_synthetic(name: str, n: int = 100, seed: int = 1) -> TensorDataset:
    """
    Build a :class:`TensorDataset` for a named problem.

    Args:
        name: one of ``Simple/Diag/Split/Xor/Circle/Spiral`` (case-insensitive).
        n: number of samples.
        seed: RNG seed (points are drawn deterministically).

    Returns:
        A ``(features (n, 2), labels (n,))`` TensorDataset.
    """
    canonical = _LOOKUP.get(name.casefold()) or name
    if canonical not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(_DATASETS)}")
    rng = random.Random(seed)
    old_state = random.getstate()
    random.setstate(rng.getstate())
    g = _DATASETS[canonical](n)
    random.setstate(old_state)

    X = from_numpy(np.asarray(g.X, dtype=np.float32))
    y = from_numpy(np.asarray(g.y, dtype=np.float32))
    return TensorDataset(X, y)


def synthetic_classification(name: str, n: int = 100, seed: int = 1) -> Graph:
    """Return the raw :class:`Graph` for a named problem (tiny helper)."""
    canonical = _LOOKUP.get(name.casefold()) or name
    if canonical not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(_DATASETS)}")
    random.seed(seed)
    return _DATASETS[name](n)


__all__ = [
    "Graph",
    "simple",
    "diag",
    "split",
    "xor",
    "circle",
    "spiral",
    "make_spiral",
    "make_synthetic",
    "synthetic_classification",
]