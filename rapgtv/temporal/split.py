"""Deterministic chronological temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _readonly_indices(start: int, stop: int) -> NDArray[np.int_]:
    result = np.arange(start, stop, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Exhaustive ordered Train/Validation/Test epoch partition.

    End times are inclusive. ``test_start_time`` is the first Test epoch.
    Arrays contain zero-based positions into the exact ``times`` used to build
    the split.
    """

    train_idx: NDArray[np.int_]
    val_idx: NDArray[np.int_]
    test_idx: NDArray[np.int_]
    train_end_time: Any
    val_end_time: Any
    test_start_time: Any
    n_epochs: int
    fractions: tuple[float, float, float]

    @property
    def train_val_idx(self) -> NDArray[np.int_]:
        values = np.concatenate((self.train_idx, self.val_idx))
        values.setflags(write=False)
        return values


def build_temporal_split(
    times: NDArray[Any],
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> TemporalSplit:
    """Build the shared chronological split using frozen period-size rules.

    Train and Validation sizes are independently floored. Test receives the
    remainder: ``n_train=floor(T*train)``, ``n_val=floor(T*validation)``, and
    ``n_test=T-n_train-n_val``. Empty periods fail closed.
    """

    times_array = np.asarray(times)
    if times_array.ndim != 1 or times_array.size < 3:
        raise ValueError("at least three one-dimensional epochs are required")
    fractions = np.asarray((train_fraction, val_fraction, test_fraction), dtype=float)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0):
        raise ValueError("split fractions must be finite and positive")
    if not np.isclose(float(fractions.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("split fractions must sum to 1")
    if times_array.dtype.kind == "f" and not np.all(np.isfinite(times_array)):
        raise ValueError("times must be finite")
    if not np.all(times_array[1:] > times_array[:-1]):
        raise ValueError("times must be strictly increasing")

    n = int(times_array.size)
    n_train = int(np.floor(n * train_fraction))
    n_val = int(np.floor(n * val_fraction))
    n_test = n - n_train - n_val
    train_stop = n_train
    val_stop = n_train + n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("fractions produce an empty Train, Validation, or Test period")

    train_idx = _readonly_indices(0, train_stop)
    val_idx = _readonly_indices(train_stop, val_stop)
    test_idx = _readonly_indices(val_stop, n)
    return TemporalSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_end_time=times_array[train_idx[-1]],
        val_end_time=times_array[val_idx[-1]],
        test_start_time=times_array[test_idx[0]],
        n_epochs=n,
        fractions=(float(train_fraction), float(val_fraction), float(test_fraction)),
    )
