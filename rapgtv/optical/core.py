"""Sentinel-2 mathematical core without raster I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapgtv.representation.scaling import RobustScaler
from rapgtv.temporal.causality import ConstructionContext
from rapgtv.data.adapters import OpticalSourceAudit, validate_optical_source


@dataclass(frozen=True, slots=True)
class OpticalCompositeSeries:
    """Masked monthly indices plus the acquisitions behind every composite."""

    values: NDArray[np.float64]
    valid_proportion: NDArray[np.float64]
    composite_times: NDArray[Any]
    source_acquisition_times: tuple[NDArray[Any], ...]
    source_ids: tuple[tuple[str, ...], ...]
    source_audit: OpticalSourceAudit | None = None

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        valid = np.array(self.valid_proportion, dtype=np.float64, copy=True)
        times = np.array(self.composite_times, copy=True)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("optical values must have shape (N, M, 3) ordered NDVI/NDMI/BSI")
        if valid.shape != values.shape[:2] or not np.all(np.isfinite(valid)):
            raise ValueError("valid_proportion must be finite with shape (N, M)")
        if np.any((valid < 0) | (valid > 1)):
            raise ValueError("valid_proportion must lie in [0, 1]")
        if times.ndim != 1 or times.size != values.shape[1] or not np.all(times[1:] > times[:-1]):
            raise ValueError("composite_times must be one strictly increasing time per month")
        if len(self.source_acquisition_times) != times.size or len(self.source_ids) != times.size:
            raise ValueError("each monthly composite requires source acquisition provenance")
        frozen_sources = []
        for month, (source_times, ids) in enumerate(zip(self.source_acquisition_times, self.source_ids)):
            source_array = np.array(source_times, copy=True)
            if source_array.ndim != 1 or source_array.size != len(ids):
                raise ValueError(f"composite {month} needs matching source times and IDs")
            if source_array.size == 0 and (
                self.source_audit is None or self.source_audit.backend_process_graph is None
            ):
                raise ValueError(f"composite {month} needs scene provenance or backend process-graph proof")
            if any(not str(identifier).strip() for identifier in ids):
                raise ValueError("source acquisition IDs must be non-empty")
            source_array.setflags(write=False)
            frozen_sources.append(source_array)
        values.setflags(write=False)
        valid.setflags(write=False)
        times.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_proportion", valid)
        object.__setattr__(self, "composite_times", times)
        object.__setattr__(self, "source_acquisition_times", tuple(frozen_sources))


@dataclass(frozen=True, slots=True)
class OpticalRepresentation:
    z_o: NDArray[np.float64]
    q_o: NDArray[np.float64]
    scaler: RobustScaler
    fit_stage: str
    fit_cutoff: Any
    composite_times: NDArray[Any]
    source_ids: tuple[tuple[str, ...], ...]
    audit_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpticalGate:
    distances: NDArray[np.float64]
    sigma_o: float
    similarity: NDArray[np.float64]
    edge_reliability: NDArray[np.float64]
    gate: NDArray[np.float64]
    audit_flags: tuple[str, ...]


def compute_optical_indices(
    b02: ArrayLike,
    b04: ArrayLike,
    b08: ArrayLike,
    b11: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute NDVI, NDMI, and BSI; zero denominators become missing."""

    bands = [np.asarray(x, dtype=np.float64) for x in (b02, b04, b08, b11)]
    if any(x.shape != bands[0].shape for x in bands):
        raise ValueError("all Sentinel-2 bands must have identical shapes")
    blue, red, nir, swir = bands

    def ratio(numerator: NDArray[np.float64], denominator: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.full_like(numerator, np.nan, dtype=np.float64)
        usable = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > epsilon)
        result[usable] = numerator[usable] / denominator[usable]
        return result

    return np.stack(
        (
            ratio(nir - red, nir + red),
            ratio(nir - swir, nir + swir),
            ratio((swir + red) - (nir + blue), (swir + red) + (nir + blue)),
        ),
        axis=-1,
    )


def _optical_features(values: NDArray[np.float64], valid: NDArray[np.float64]) -> NDArray[np.float64]:
    n_nodes = values.shape[0]
    features = np.full((n_nodes, 15), np.nan, dtype=np.float64)
    for node in range(n_nodes):
        for index in range(3):
            usable = (valid[node] > 0) & np.isfinite(values[node, :, index])
            if not np.any(usable):
                continue
            series = values[node, usable, index]
            offset = index * 5
            features[node, offset : offset + 5] = (
                np.median(series),
                np.std(series, ddof=0),
                np.quantile(series, 0.25),
                np.quantile(series, 0.75),
                series[-1] - series[0],
            )
    return features


def build_optical_representation(
    composites: OpticalCompositeSeries,
    context: ConstructionContext,
    *,
    epsilon: float = 1e-8,
) -> OpticalRepresentation:
    """Build 15-D optical features only after auditing source acquisitions."""

    context.assert_times_allowed(composites.composite_times, source="optical composites")
    if composites.source_audit is not None:
        validate_optical_source(composites.source_audit, context)
    for month, source_times in enumerate(composites.source_acquisition_times):
        if source_times.size:
            context.assert_times_allowed(source_times, source=f"optical composite {month} acquisitions")

    q_o = np.mean(composites.valid_proportion, axis=1)
    raw = _optical_features(composites.values, composites.valid_proportion)
    flags = []
    for column in range(raw.shape[1]):
        missing = ~np.isfinite(raw[:, column])
        if not np.any(missing):
            continue
        available = raw[~missing, column]
        fill = float(np.median(available)) if available.size else 0.0
        if not available.size:
            flags.append(f"all_missing_optical_feature_{column}")
        raw[missing, column] = fill
    scaler = RobustScaler(epsilon).fit(raw)
    z_o = scaler.transform(raw)
    return OpticalRepresentation(
        z_o,
        q_o,
        scaler,
        context.stage,
        context.cutoff_time,
        composites.composite_times,
        composites.source_ids,
        tuple(flags),
    )


def compute_optical_gate(
    z_o: ArrayLike,
    q_o: ArrayLike,
    edge_index: ArrayLike,
    *,
    epsilon: float = 1e-8,
) -> OpticalGate:
    """Compute Step-4 optical distance, similarity, reliability, and gate."""

    features = np.asarray(z_o, dtype=np.float64)
    reliability = np.asarray(q_o, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("z_o must be a finite two-dimensional array")
    if reliability.shape != (features.shape[0],) or np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("q_o must have shape (N,) and lie in [0, 1]")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        raise ValueError("edge_index must be a non-empty E by 2 array")
    left, right = edges[:, 0], edges[:, 1]
    distances = np.linalg.norm(features[left] - features[right], axis=1)
    raw_sigma = float(np.median(distances))
    sigma_o = max(raw_sigma, epsilon)
    similarity = np.exp(np.maximum(-(distances**2) / (2.0 * sigma_o**2), np.log(np.finfo(float).tiny)))
    edge_reliability = np.sqrt(reliability[left] * reliability[right])
    gate = np.clip(1.0 - edge_reliability * (1.0 - similarity), 0.0, 1.0)
    flags = ("degenerate_optical_scale",) if raw_sigma <= epsilon else ()
    return OpticalGate(distances, sigma_o, similarity, edge_reliability, gate, flags)
