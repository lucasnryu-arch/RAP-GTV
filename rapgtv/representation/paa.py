"""Canonical piecewise aggregate approximation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


PAA_LENGTH = 12


def paa(values: ArrayLike, n_segments: int = PAA_LENGTH) -> NDArray[np.float64]:
    """Return equal-width PAA using fractional weights at bin boundaries.

    The input is interpreted as a step function with one unit-width cell per
    ordered acquisition. Fractional boundary weights avoid an arbitrary
    ``array_split`` imbalance when the number of epochs is not divisible by 12.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("PAA input must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("PAA input must be finite")
    if n_segments != PAA_LENGTH:
        raise ValueError("canonical PAA length is frozen at 12")
    n_times = array.shape[1]
    if n_times < n_segments:
        raise ValueError("PAA12 requires at least 12 temporal values")

    boundaries = np.linspace(0.0, float(n_times), n_segments + 1)
    output = np.empty((array.shape[0], n_segments), dtype=np.float64)
    cells = np.arange(n_times, dtype=np.float64)
    for segment in range(n_segments):
        left, right = boundaries[segment], boundaries[segment + 1]
        weights = np.maximum(0.0, np.minimum(cells + 1.0, right) - np.maximum(cells, left))
        output[:, segment] = array @ weights / (right - left)
    return output
