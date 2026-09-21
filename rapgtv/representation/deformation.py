"""Frozen three-block causal deformation representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapgtv.data.schema import SiteDataset
from rapgtv.temporal.causality import ConstructionContext

from .paa import PAA_LENGTH, paa
from .scaling import RobustScaler


NUMERICAL_EPSILON = 1e-8


def _as_years(times: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(times)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two chronological times are required")
    if values.dtype.kind == "M":
        days = (values.astype("datetime64[ns]") - values[0].astype("datetime64[ns]")) / np.timedelta64(1, "D")
        years = np.asarray(days, dtype=np.float64) / 365.25
    elif values.dtype.kind == "m":
        days = (values - values[0]) / np.timedelta64(1, "D")
        years = np.asarray(days, dtype=np.float64) / 365.25
    elif values.dtype.kind in "fiu":
        years = values.astype(np.float64) - float(values[0])
    else:
        raise TypeError("times must be datetime, timedelta, integer, or floating values")
    if not np.all(np.isfinite(years)) or not np.all(np.diff(years) > 0):
        raise ValueError("times must be finite and strictly increasing")
    return years


def build_working_displacement(
    dataset: SiteDataset,
    context: ConstructionContext,
) -> NDArray[np.float64]:
    """Causally interpolate a complete working series without changing validity.

    Interpolation uses actual acquisition time. Interior gaps are linear and
    boundary gaps use the nearest observed value. A wholly unobserved node is
    rejected because its deformation representation is undefined.
    """

    displacement = np.asarray(context.slice_time_axis(dataset.displacement), dtype=np.float64)
    valid = np.asarray(context.slice_time_axis(dataset.original_valid_mask), dtype=bool)
    times = _as_years(context.slice_time_axis(dataset.times, axis=0))
    working = np.empty_like(displacement)
    for node in range(dataset.n_points):
        observed = valid[node]
        if not np.any(observed):
            raise ValueError(f"node {node} has no originally valid observation before the cutoff")
        working[node] = np.interp(times, times[observed], displacement[node, observed])
    if not np.all(np.isfinite(working)):
        raise RuntimeError("causal working displacement contains non-finite values")
    return working


def compute_time_aware_rate(
    displacement: ArrayLike,
    times: ArrayLike,
) -> NDArray[np.float64]:
    """Compute displacement per year with each true acquisition interval."""

    values = np.asarray(displacement, dtype=np.float64)
    years = _as_years(times)
    if values.ndim != 2 or values.shape[1] != years.size:
        raise ValueError("displacement and times have incompatible shapes")
    if not np.all(np.isfinite(values)):
        raise ValueError("time-aware rate requires finite working displacement")
    return np.diff(values, axis=1) / np.diff(years)[None, :]


def _row_robust_normalize(values: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    return (values - median) / (mad + epsilon)


def long_term_statistics(
    displacement: ArrayLike,
    times: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the ordered 14-dimensional Step-4 long-term block."""

    values = np.asarray(displacement, dtype=np.float64)
    years = _as_years(times)
    if values.ndim != 2 or values.shape[1] != years.size or values.shape[1] < 2:
        raise ValueError("long-term statistics require an N by T array with T >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("long-term statistics require finite working displacement")

    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1, ddof=0)
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None]), axis=1)
    quantiles = np.quantile(values, (0.05, 0.25, 0.75, 0.95), axis=1).T

    design_trend = np.column_stack((np.ones_like(years), years))
    trend = np.linalg.lstsq(design_trend, values.T, rcond=None)[0][1]
    rate = compute_time_aware_rate(values, times)
    mean_abs_rate = np.mean(np.abs(rate), axis=1)
    max_abs_rate = np.max(np.abs(rate), axis=1)
    end_start = values[:, -1] - values[:, 0]

    midpoint = values.shape[1] // 2
    late_early = np.mean(values[:, midpoint:], axis=1) - np.mean(values[:, :midpoint], axis=1)

    annual_design = np.column_stack(
        (np.ones_like(years), years, np.sin(2.0 * np.pi * years), np.cos(2.0 * np.pi * years))
    )
    annual_coefficients = np.linalg.lstsq(annual_design, values.T, rcond=None)[0]
    annual_amplitude = np.hypot(annual_coefficients[2], annual_coefficients[3])

    return np.column_stack(
        (
            mean,
            std,
            median,
            mad,
            quantiles,
            trend,
            mean_abs_rate,
            max_abs_rate,
            end_start,
            late_early,
            annual_amplitude,
        )
    )


def _canonicalize_component_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    result = components.copy()
    for row in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0:
            result[row] *= -1.0
    return result


@dataclass(frozen=True, slots=True)
class DeformationRepresentation:
    z_d: NDArray[np.float64]
    components: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    p_star: int
    block_scalers: tuple[RobustScaler, RobustScaler, RobustScaler]
    block_energies: tuple[float, float, float]
    latent_scaler: RobustScaler
    fit_stage: str
    fit_cutoff: Any


def build_deformation_representation(
    dataset: SiteDataset,
    context: ConstructionContext,
    *,
    epsilon: float = NUMERICAL_EPSILON,
) -> DeformationRepresentation:
    """Build ``Z_D`` entirely from the construction context."""

    working = build_working_displacement(dataset, context)
    times = context.slice_time_axis(dataset.times, axis=0)
    rate = compute_time_aware_rate(working, times)
    if working.shape[1] < PAA_LENGTH or rate.shape[1] < PAA_LENGTH:
        raise ValueError("the construction period needs at least 13 epochs for both PAA12 blocks")

    raw_blocks = (
        long_term_statistics(working, times),
        paa(_row_robust_normalize(working, epsilon)),
        paa(_row_robust_normalize(rate, epsilon)),
    )
    scalers = tuple(RobustScaler(epsilon).fit(block) for block in raw_blocks)
    scaled_blocks = tuple(scaler.transform(block) for scaler, block in zip(scalers, raw_blocks))
    energies = tuple(float(np.sqrt(np.sum(block * block) / block.shape[0])) for block in scaled_blocks)
    normalized_blocks = tuple(block / max(energy, epsilon) for block, energy in zip(scaled_blocks, energies))
    deformation_features = np.concatenate(normalized_blocks, axis=1) / np.sqrt(3.0)

    if deformation_features.shape[0] < 2:
        raise ValueError("PCA requires at least two nodes")
    pca_mean = np.mean(deformation_features, axis=0)
    centered = deformation_features - pca_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = singular_values**2 / (deformation_features.shape[0] - 1)
    total_variance = float(np.sum(explained_variance))
    if total_variance <= epsilon:
        p_star = 1
        ratios = np.zeros_like(explained_variance)
        ratios[0] = 1.0
    else:
        ratios = explained_variance / total_variance
        p_star = int(np.searchsorted(np.cumsum(ratios), 0.95, side="left") + 1)
    components = _canonicalize_component_signs(vt[:p_star])
    latent = centered @ components.T
    latent_scaler = RobustScaler(epsilon).fit(latent)
    z_d = latent_scaler.transform(latent)

    return DeformationRepresentation(
        z_d=z_d,
        components=components,
        pca_mean=pca_mean,
        explained_variance=explained_variance,
        explained_variance_ratio=ratios,
        p_star=p_star,
        block_scalers=scalers,  # type: ignore[arg-type]
        block_energies=energies,  # type: ignore[arg-type]
        latent_scaler=latent_scaler,
        fit_stage=context.stage,
        fit_cutoff=context.cutoff_time,
    )
