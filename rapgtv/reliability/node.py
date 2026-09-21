"""Step-4 InSAR reliability components and fidelity budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapgtv.data.schema import SiteDataset
from rapgtv.temporal.causality import ConstructionContext


Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True, slots=True)
class QualityMetadata:
    """A site-adapter-validated node quality variable with explicit direction."""

    values: NDArray[np.float64]
    direction: Direction


@dataclass(frozen=True, slots=True)
class NodeReliability:
    q_meta: NDArray[np.float64]
    q_valid: NDArray[np.float64]
    q_res: NDArray[np.float64]
    residual_scale: NDArray[np.float64]
    q_d: NDArray[np.float64]
    rho: NDArray[np.float64]
    residual_reliability_degenerate: bool
    fit_stage: str
    fit_cutoff: object


@dataclass(frozen=True, slots=True)
class ResidualReliability:
    q_res: NDArray[np.float64]
    residual_reliability_degenerate: bool


def _q02_q98_map(values: ArrayLike, *, higher_is_better: bool, epsilon: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("reliability metadata/residuals must be finite one-dimensional arrays")
    q02, q98 = np.quantile(array, (0.02, 0.98))
    scaled = np.clip((array - q02) / (q98 - q02 + epsilon), 0.0, 1.0)
    return scaled if higher_is_better else 1.0 - scaled


def compute_metadata_reliability(
    metadata: Mapping[str, QualityMetadata],
    n_nodes: int,
    *,
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Map only explicit, directed metadata and combine by geometric mean."""

    if not metadata:
        return np.ones(n_nodes, dtype=np.float64)
    mapped = []
    for name, item in metadata.items():
        if not name:
            raise ValueError("quality metadata names must be non-empty")
        values = np.asarray(item.values, dtype=np.float64)
        if values.shape != (n_nodes,):
            raise ValueError(f"quality metadata {name!r} must have shape ({n_nodes},)")
        if item.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(f"quality metadata {name!r} has no valid direction")
        mapped.append(
            _q02_q98_map(
                values,
                higher_is_better=item.direction == "higher_is_better",
                epsilon=epsilon,
            )
        )
    return np.prod(np.stack(mapped, axis=0), axis=0) ** (1.0 / len(mapped))


def compute_residual_reliability(
    residual_scale: ArrayLike,
    *,
    epsilon: float = 1e-8,
) -> ResidualReliability:
    """Map defined residual scales; undefined nodes abstain with ``q_res=1``."""

    residual = np.asarray(residual_scale, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0 or np.any(np.isinf(residual)):
        raise ValueError("residual_scale must be a non-empty vector containing finite values or NaN")
    defined = np.isfinite(residual)
    q_res = np.ones(residual.size, dtype=np.float64)
    # Fewer than two nodes cannot define an across-node ordering. The same
    # abstention applies when Q02 and Q98 are numerically indistinguishable.
    if np.count_nonzero(defined) < 2:
        return ResidualReliability(q_res, True)
    q02, q98 = np.quantile(residual[defined], (0.02, 0.98))
    if q98 - q02 <= epsilon:
        return ResidualReliability(q_res, True)
    q_res[defined] = 1.0 - np.clip(
        (residual[defined] - q02) / (q98 - q02 + epsilon),
        0.0,
        1.0,
    )
    return ResidualReliability(q_res, False)


def compute_local_residuals(
    dataset: SiteDataset,
    context: ConstructionContext,
) -> NDArray[np.float64]:
    """Return residuals from the fixed centered five-valid-observation median.

    Each valid observation uses itself, up to two preceding valid observations,
    and up to two following valid observations. Boundary windows shorten
    naturally. Invalid/insufficient epochs remain NaN and never enter the node
    residual median.
    """

    displacement = np.asarray(context.slice_time_axis(dataset.displacement), dtype=np.float64)
    valid = np.asarray(context.slice_time_axis(dataset.original_valid_mask), dtype=bool)
    residuals = np.full(displacement.shape, np.nan, dtype=np.float64)
    for node in range(dataset.n_points):
        valid_positions = np.flatnonzero(valid[node])
        if valid_positions.size < 3:
            continue
        observed = displacement[node, valid_positions]
        for rank, position in enumerate(valid_positions):
            window = observed[max(0, rank - 2) : min(observed.size, rank + 3)]
            if window.size >= 3:
                residuals[node, position] = abs(observed[rank] - np.median(window))
    return residuals


def compute_local_residual_scale(
    dataset: SiteDataset,
    context: ConstructionContext,
) -> NDArray[np.float64]:
    """Return each node's median eligible local residual, or NaN to abstain."""

    residuals = compute_local_residuals(dataset, context)
    scale = np.full(dataset.n_points, np.nan, dtype=np.float64)
    for node in range(dataset.n_points):
        eligible = np.isfinite(residuals[node])
        if np.any(eligible):
            scale[node] = np.median(residuals[node, eligible])
    return scale


def compute_node_reliability(
    dataset: SiteDataset,
    context: ConstructionContext,
    quality_metadata: Mapping[str, QualityMetadata],
    *,
    epsilon: float = 1e-8,
    epsilon_d: float = 1e-8,
) -> NodeReliability:
    """Compute all frozen reliability terms within one construction context."""

    q_meta = compute_metadata_reliability(quality_metadata, dataset.n_points, epsilon=epsilon)
    original_valid = np.asarray(context.slice_time_axis(dataset.original_valid_mask), dtype=bool)
    q_valid = np.mean(original_valid, axis=1, dtype=np.float64)
    residual_scale = compute_local_residual_scale(dataset, context)
    residual_result = compute_residual_reliability(residual_scale, epsilon=epsilon)
    q_res = residual_result.q_res
    q_d = np.cbrt(np.clip(q_meta * q_valid * q_res, 0.0, 1.0))
    raw = q_d + epsilon_d
    rho = raw / np.mean(raw)
    if not np.isclose(np.mean(rho), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("node fidelity budget normalization failed")
    return NodeReliability(
        q_meta,
        q_valid,
        q_res,
        residual_scale,
        q_d,
        rho,
        residual_result.residual_reliability_degenerate,
        context.stage,
        context.cutoff_time,
    )
