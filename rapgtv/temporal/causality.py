"""Explicit temporal construction boundaries for future scientific modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .split import TemporalSplit


ConstructionStage = Literal["train", "train_val"]


@dataclass(frozen=True, slots=True)
class ConstructionContext:
    """The only epochs and latest time allowed for model construction."""

    stage: ConstructionStage
    allowed_indices: NDArray[np.int_]
    cutoff_time: Any
    split: TemporalSplit

    def assert_indices_allowed(self, indices: NDArray[Any], *, source: str = "construction") -> None:
        """Reject positions outside this context, including all Test epochs."""

        candidate = np.asarray(indices)
        if candidate.ndim != 1 or candidate.dtype.kind not in "iu":
            raise TypeError(f"{source} indices must be a one-dimensional integer array")
        if np.any(candidate < 0) or np.any(candidate >= self.split.n_epochs):
            raise ValueError(f"{source} indices are outside the temporal axis")
        if not np.all(np.isin(candidate, self.allowed_indices)):
            raise ValueError(f"{source} contains epochs after the {self.stage} cutoff")

    def assert_times_allowed(self, times: NDArray[Any], *, source: str = "construction") -> None:
        """Reject observations/acquisitions later than the inclusive cutoff."""

        candidate = np.asarray(times)
        if candidate.ndim != 1:
            raise ValueError(f"{source} times must be one-dimensional")
        if candidate.size and np.any(candidate > self.cutoff_time):
            raise ValueError(f"{source} contains times after the {self.stage} cutoff")

    def slice_time_axis(self, values: NDArray[Any], axis: int = -1) -> NDArray[Any]:
        """Return an allowed construction view from a full temporal array."""

        array = np.asarray(values)
        normalized_axis = axis if axis >= 0 else array.ndim + axis
        if normalized_axis < 0 or normalized_axis >= array.ndim:
            raise ValueError(f"axis {axis} is out of bounds for an array of dimension {array.ndim}")
        if array.shape[normalized_axis] != self.split.n_epochs:
            raise ValueError("selected axis length does not match the split temporal axis")
        return np.take(array, self.allowed_indices, axis=normalized_axis)


def build_construction_context(
    split: TemporalSplit,
    stage: ConstructionStage = "train",
) -> ConstructionContext:
    """Create the validation-selection or final-refit information boundary."""

    if stage == "train":
        indices = split.train_idx
        cutoff = split.train_end_time
    elif stage == "train_val":
        indices = split.train_val_idx
        cutoff = split.val_end_time
    else:
        raise ValueError("stage must be 'train' or 'train_val'")
    frozen = np.array(indices, dtype=np.int64, copy=True)
    frozen.setflags(write=False)
    return ConstructionContext(stage=stage, allowed_indices=frozen, cutoff_time=cutoff, split=split)


def get_allowed_indices(split: TemporalSplit, stage: ConstructionStage = "train") -> NDArray[np.int_]:
    """Return immutable construction indices for the requested stage."""

    return build_construction_context(split, stage).allowed_indices
