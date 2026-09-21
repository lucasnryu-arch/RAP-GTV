"""Frozen future kinematic consistency metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_fdd(displacement: ArrayLike, labels: ArrayLike, test_indices: ArrayLike) -> float:
    """Compute Test-start-rebased future deformation dispersion."""

    values = np.asarray(displacement, dtype=np.float64)
    clusters = np.asarray(labels)
    indices = np.asarray(test_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("FDD displacement must be an N by T array")
    if clusters.ndim != 1 or clusters.shape[0] != values.shape[0]:
        raise ValueError("FDD labels must have shape (N,)")
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("test_indices must be a non-empty valid index vector")
    if not np.all(indices[1:] > indices[:-1]):
        raise ValueError("test_indices must be strictly increasing")
    future = values[:, indices]
    if not np.all(np.isfinite(future)):
        raise ValueError("FDD requires finite observations throughout the selected Test period")
    rebased = future - future[:, [0]]
    total = 0.0
    count = 0
    for label in np.unique(clusters):
        members = rebased[clusters == label]
        median_curve = np.median(members, axis=0)
        total += float(np.sum(np.abs(members - median_curve)))
        count += int(members.size)
    return float(total / count)
