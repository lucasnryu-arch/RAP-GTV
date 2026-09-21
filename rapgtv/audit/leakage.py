"""Fail-closed temporal and inventory leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rapgtv.temporal.causality import ConstructionContext
from rapgtv.temporal.split import TemporalSplit


class LeakageError(ValueError):
    """Raised when information crosses a frozen scientific boundary."""


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    """Successful auditable result for a checked construction operation."""

    stage: str
    source: str
    cutoff_time: Any
    checked_count: int


def audit_temporal_split(split: TemporalSplit, times: NDArray[Any]) -> None:
    """Verify exhaustive, ordered, disjoint indices and recorded endpoints."""

    time_values = np.asarray(times)
    if time_values.ndim != 1 or time_values.size != split.n_epochs:
        raise LeakageError("split and times length disagree")
    groups = (split.train_idx, split.val_idx, split.test_idx)
    if any(group.size == 0 for group in groups):
        raise LeakageError("Train, Validation, and Test must all be non-empty")
    combined = np.concatenate(groups)
    if not np.array_equal(combined, np.arange(split.n_epochs)):
        raise LeakageError("split must be chronological, exhaustive, and non-overlapping")
    if not (np.all(split.train_idx < split.val_idx[0]) and np.all(split.val_idx < split.test_idx[0])):
        raise LeakageError("split periods overlap or are out of order")
    if split.train_end_time != time_values[split.train_idx[-1]]:
        raise LeakageError("train_end_time disagrees with times")
    if split.val_end_time != time_values[split.val_idx[-1]]:
        raise LeakageError("val_end_time disagrees with times")
    if split.test_start_time != time_values[split.test_idx[0]]:
        raise LeakageError("test_start_time disagrees with times")
    if not (split.train_end_time < time_values[split.val_idx[0]] <= split.val_end_time < split.test_start_time):
        raise LeakageError("recorded time boundaries are not strictly chronological")


def audit_construction_indices(
    context: ConstructionContext,
    indices: NDArray[Any],
    *,
    source: str,
) -> LeakageAudit:
    """Audit deformation, quality, transform, or other construction indices."""

    try:
        context.assert_indices_allowed(indices, source=source)
    except (TypeError, ValueError) as exc:
        raise LeakageError(str(exc)) from exc
    return LeakageAudit(context.stage, source, context.cutoff_time, int(np.asarray(indices).size))


def audit_acquisition_times(
    context: ConstructionContext,
    times: NDArray[Any],
    *,
    source: str = "optical acquisition",
) -> LeakageAudit:
    """Audit source-scene dates before any future compositing operation."""

    try:
        context.assert_times_allowed(times, source=source)
    except ValueError as exc:
        raise LeakageError(str(exc)) from exc
    return LeakageAudit(context.stage, source, context.cutoff_time, int(np.asarray(times).size))


def audit_inventory_use(
    *,
    used: bool,
    purpose: Literal["evaluation", "model_construction", "selection"],
) -> None:
    """Permit inventory only for downstream evaluation, never construction."""

    if used and purpose != "evaluation":
        raise LeakageError(f"inventory cannot be used for {purpose}")

