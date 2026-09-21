"""Frozen mechanism-level generator for controlled Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rapgtv.representation.scaling import RobustScaler


GRID_SIDE: Final = 20
N_NODES: Final = GRID_SIDE * GRID_SIDE
LATENT_DIM: Final = 3
WITHIN_REGIME_AMPLITUDE: Final = 0.15
SIGMA_GOOD: Final = 0.05
LOW_QUALITY_FRACTION: Final = 0.25
OPTICAL_FEATURE_NAMES: Final = tuple(
    f"{index}_{statistic}"
    for index in ("NDVI", "NDMI", "BSI")
    for statistic in ("median", "std", "q25", "q75", "last_minus_first")
)


@dataclass(frozen=True, slots=True)
class ControlledCondition:
    """One controlled world; factors have disjoint scientific roles."""

    h: float
    eta_t: int
    eta_o: int
    jump: float
    q_o: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.h) or self.h < 1:
            raise ValueError("H must be finite and at least one")
        if self.eta_t not in (0, 1, 2) or self.eta_o not in (0, 1, 2):
            raise ValueError("eta_T and eta_O must be in {0, 1, 2}")
        if not np.isfinite(self.jump) or self.jump <= 0:
            raise ValueError("J must be finite and positive")
        if not np.isfinite(self.q_o) or not 0 <= self.q_o <= 1:
            raise ValueError("q_O must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ControlledRealization:
    """Separated generator truth, observations, and canonical model inputs."""

    seed: int
    condition: ControlledCondition
    coordinates: NDArray[np.float64]
    true_labels: NDArray[np.int64]
    u_star: NDArray[np.float64]
    z_d: NDArray[np.float64]
    standard_normal_noise: NDArray[np.float64]
    observation_sigma: NDArray[np.float64]
    low_quality_mask: NDArray[np.bool_]
    q_d: NDArray[np.float64]
    rho: NDArray[np.float64]
    elevation: NDArray[np.float64]
    slope: NDArray[np.float64]
    aspect_radians: NDArray[np.float64]
    curvature: NDArray[np.float64]
    tri: NDArray[np.float64]
    optical_raw: NDArray[np.float64]
    z_o: NDArray[np.float64]
    q_o: NDArray[np.float64]
    evaluation_edges: NDArray[np.int64]
    optical_feature_names: tuple[str, ...] = OPTICAL_FEATURE_NAMES


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent streams keep factor changes paired within a seed."""

    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def regular_grid() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    rows, columns = np.meshgrid(np.arange(GRID_SIDE), np.arange(GRID_SIDE), indexing="ij")
    coords = np.column_stack((columns.ravel(), rows.ravel())).astype(np.float64)
    return coords, rows.ravel().astype(np.int64), columns.ravel().astype(np.int64)


def three_strip_labels(columns: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return fixed contiguous strip sizes 140, 120, and 140."""

    labels = np.zeros(columns.size, dtype=np.int64)
    labels[columns >= 7] = 1
    labels[columns >= 13] = 2
    return labels


def lattice_four_neighbor_edges() -> NDArray[np.int64]:
    """Evaluation-only graph, independent of every model graph."""

    edges: list[tuple[int, int]] = []
    for row in range(GRID_SIDE):
        for column in range(GRID_SIDE):
            node = row * GRID_SIDE + column
            if column + 1 < GRID_SIDE:
                edges.append((node, node + 1))
            if row + 1 < GRID_SIDE:
                edges.append((node, node + GRID_SIDE))
    return np.asarray(edges, dtype=np.int64)


def _within_regime_field(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    labels: NDArray[np.int64],
    seed: int,
) -> NDArray[np.float64]:
    rng = _rng(seed, 11)
    field = np.empty((N_NODES, LATENT_DIM), dtype=np.float64)
    starts = (0, 7, 13)
    widths = (7, 6, 7)
    y = rows / (GRID_SIDE - 1)
    for regime in range(3):
        mask = labels == regime
        local_x = (columns[mask] - starts[regime]) / max(widths[regime] - 1, 1)
        yy = y[mask]
        basis = np.column_stack(
            (
                np.sin(np.pi * local_x),
                np.cos(np.pi * yy),
                np.sin(np.pi * (local_x + yy) / 2.0),
                np.cos(np.pi * (local_x - yy) / 2.0),
            )
        )
        coefficients = rng.normal(size=(basis.shape[1], LATENT_DIM))
        values = basis @ coefficients
        values -= np.mean(values, axis=0, keepdims=True)
        field[mask] = values
    rms = float(np.sqrt(np.mean(np.sum(field**2, axis=1))))
    if rms <= 1e-12:
        raise RuntimeError("within-regime field is degenerate")
    return WITHIN_REGIME_AMPLITUDE * field / rms


def _latent_truth(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    labels: NDArray[np.int64],
    jump: float,
    seed: int,
) -> NDArray[np.float64]:
    within = _within_regime_field(rows, columns, labels, seed)
    direction = np.asarray([1.0, -0.5, 0.25], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    centroids = (np.arange(3, dtype=np.float64) - 1.0)[:, None] * float(jump) * direction
    return centroids[labels] + within


def _low_quality_patch(rows: NDArray[np.int64], columns: NDArray[np.int64]) -> NDArray[np.bool_]:
    mask = (rows >= 5) & (rows < 15) & (columns >= 5) & (columns < 15)
    if int(np.count_nonzero(mask)) != int(N_NODES * LOW_QUALITY_FRACTION):
        raise RuntimeError("low-quality patch does not contain exactly 25% of nodes")
    return mask


def _terrain_fields(
    rows: NDArray[np.int64],
    labels: NDArray[np.int64],
    eta_t: int,
    seed: int,
) -> tuple[NDArray[np.float64], ...]:
    """Create label-neutral y-profiles plus eta_T-scaled regime contrast."""

    rng = _rng(seed, 23)
    y = np.arange(GRID_SIDE, dtype=np.float64) / (GRID_SIDE - 1)
    row_noise = rng.normal(size=(GRID_SIDE, 5))
    elevation_row = 100.0 + 5.0 * np.sin(2 * np.pi * y) + 0.5 * row_noise[:, 0]
    slope_row = 10.0 + 2.0 * np.cos(2 * np.pi * y) + 0.2 * row_noise[:, 1]
    aspect_row = np.mod(0.8 + 0.5 * np.sin(2 * np.pi * y + 0.3) + 0.04 * row_noise[:, 2], 2 * np.pi)
    curvature_row = 0.08 * np.sin(4 * np.pi * y) + 0.01 * row_noise[:, 3]
    tri_row = 2.0 + 0.35 * np.cos(2 * np.pi * y - 0.2) + 0.04 * row_noise[:, 4]
    contrast = (labels.astype(np.float64) - 1.0) * float(eta_t)
    return (
        elevation_row[rows] + 8.0 * contrast,
        slope_row[rows] + 1.2 * contrast,
        aspect_row[rows],
        curvature_row[rows] + 0.025 * contrast,
        tri_row[rows] + 0.18 * contrast,
    )


def _optical_features(
    rows: NDArray[np.int64],
    labels: NDArray[np.int64],
    eta_o: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate 15-D NDVI/NDMI/BSI summary semantics before canonical scaling."""

    rng = _rng(seed, 37)
    base = np.asarray(
        [
            0.45, 0.08, 0.35, 0.55, 0.02,
            0.25, 0.07, 0.17, 0.33, -0.01,
            -0.05, 0.09, -0.15, 0.05, 0.01,
        ],
        dtype=np.float64,
    )
    row_latent = rng.normal(size=(GRID_SIDE, 4))
    loadings = rng.normal(scale=0.018, size=(4, 15))
    row_profile = base[None, :] + row_latent @ loadings
    contrast_direction = np.asarray(
        [1.0, 0.25, 0.8, 1.2, 0.35, -0.8, 0.2, -0.7, -0.9, -0.3, 0.7, 0.2, 0.6, 0.8, 0.25],
        dtype=np.float64,
    )
    raw = row_profile[rows].copy()
    raw += 0.06 * float(eta_o) * (labels.astype(np.float64) - 1.0)[:, None] * contrast_direction
    scaled = RobustScaler().fit_transform(raw)
    return raw, scaled


def generate_controlled_realization(
    condition: ControlledCondition,
    seed: int,
    *,
    epsilon_d: float = 1e-8,
) -> ControlledRealization:
    """Generate exactly one paired N=400, p=3 controlled realization."""

    coordinates, rows, columns = regular_grid()
    labels = three_strip_labels(columns)
    u_star = _latent_truth(rows, columns, labels, condition.jump, seed)
    low_quality = _low_quality_patch(rows, columns)
    standard_noise = _rng(seed, 19).normal(size=(N_NODES, LATENT_DIM))
    sigma = np.full(N_NODES, SIGMA_GOOD, dtype=np.float64)
    sigma[low_quality] *= float(condition.h)
    z_d = u_star + standard_noise * sigma[:, None]
    q_d = np.ones(N_NODES, dtype=np.float64)
    q_d[low_quality] = 1.0 / float(condition.h)
    raw_reliability = q_d + epsilon_d
    # Preserve the exact H=1 null while remaining algebraically identical to
    # the canonical mean-one normalization.
    rho = (
        np.ones(N_NODES, dtype=np.float64)
        if np.ptp(raw_reliability) == 0
        else raw_reliability / np.mean(raw_reliability)
    )
    terrain = _terrain_fields(rows, labels, condition.eta_t, seed)
    optical_raw, z_o = _optical_features(rows, labels, condition.eta_o, seed)
    q_o = np.full(N_NODES, float(condition.q_o), dtype=np.float64)
    realization = ControlledRealization(
        int(seed),
        condition,
        coordinates,
        labels,
        u_star,
        z_d,
        standard_noise,
        sigma,
        low_quality,
        q_d,
        rho,
        *terrain,
        optical_raw,
        z_o,
        q_o,
        lattice_four_neighbor_edges(),
    )
    arrays = (
        realization.coordinates,
        realization.u_star,
        realization.z_d,
        realization.rho,
        realization.elevation,
        realization.slope,
        realization.aspect_radians,
        realization.curvature,
        realization.tri,
        realization.z_o,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("controlled generator produced a non-finite model object")
    if not np.isclose(np.mean(realization.rho), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("controlled reliability budget does not have mean one")
    return realization


def regime_separability(features: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Minimum centroid distance divided by pooled within-regime RMS."""

    values = np.asarray(features, dtype=np.float64)
    centroids = np.stack([np.mean(values[labels == regime], axis=0) for regime in range(3)])
    within = values - centroids[labels]
    within_rms = max(float(np.sqrt(np.mean(np.sum(within**2, axis=1)))), 1e-12)
    adjacent = [np.linalg.norm(centroids[1] - centroids[0]), np.linalg.norm(centroids[2] - centroids[1])]
    return float(min(adjacent) / within_rms)


def terrain_matrix(realization: ControlledRealization) -> NDArray[np.float64]:
    return np.column_stack(
        (
            realization.elevation,
            realization.slope,
            np.sin(realization.aspect_radians),
            np.cos(realization.aspect_radians),
            realization.curvature,
            realization.tri,
        )
    )


def realization_diagnostics(realization: ControlledRealization) -> dict[str, object]:
    labels = realization.true_labels
    edges = realization.evaluation_edges
    boundary = labels[edges[:, 0]] != labels[edges[:, 1]]
    centroids = np.stack([np.mean(realization.u_star[labels == regime], axis=0) for regime in range(3)])
    adjacent_jump = min(np.linalg.norm(centroids[1] - centroids[0]), np.linalg.norm(centroids[2] - centroids[1]))
    return {
        "seed": realization.seed,
        "condition": {
            "H": realization.condition.h,
            "eta_T": realization.condition.eta_t,
            "eta_O": realization.condition.eta_o,
            "J": realization.condition.jump,
            "q_O": realization.condition.q_o,
        },
        "regime_sizes": np.bincount(labels, minlength=3).tolist(),
        "n_evaluation_edges": int(edges.shape[0]),
        "n_boundary_edges": int(np.count_nonzero(boundary)),
        "n_nonboundary_edges": int(np.count_nonzero(~boundary)),
        "within_regime_amplitude": WITHIN_REGIME_AMPLITUDE,
        "adjacent_centroid_jump": float(adjacent_jump),
        "observation_sigma_good": float(np.unique(realization.observation_sigma[~realization.low_quality_mask])[0]),
        "observation_sigma_bad": float(np.unique(realization.observation_sigma[realization.low_quality_mask])[0]),
        "terrain_separability": regime_separability(terrain_matrix(realization), labels),
        "optical_separability": regime_separability(realization.z_o, labels),
        "all_finite": True,
    }
