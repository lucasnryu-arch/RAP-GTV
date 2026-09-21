"""The single validation selector shared by ST, RAC, and RAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_index: int
    selected_config_id: str
    minimum_fdd: float
    threshold: float
    rows: tuple[dict[str, Any], ...]


def select_hyperparameters(
    validation_results: Sequence[Mapping[str, Any]],
    noninferiority_margin: float = 0.05,
) -> SelectionResult:
    if not validation_results:
        raise ValueError("validation_results must be non-empty")
    if noninferiority_margin != 0.05:
        raise ValueError("real-data noninferiority margin is frozen at 0.05")
    rows = [dict(row) for row in validation_results]
    if len({row["method"] for row in rows}) != 1:
        raise ValueError("select one method at a time")
    minimum = min(float(row["validation_FDD"]) for row in rows)
    threshold = 1.05 * minimum
    eligible = []
    for index, row in enumerate(rows):
        inside = float(row["validation_FDD"]) <= threshold
        row["within_noninferiority_set"] = inside
        row["selected"] = False
        if inside:
            eligible.append(index)
    selected = min(
        eligible,
        key=lambda i: (
            float(rows[i]["validation_FI"]),
            float(rows[i]["validation_FDD"]),
            rows[i].get("simplicity_rank", i),
            i,
        ),
    )
    rows[selected]["selected"] = True
    return SelectionResult(selected, str(rows[selected]["config_id"]), minimum, threshold, tuple(rows))
