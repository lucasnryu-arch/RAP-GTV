"""Deterministic terrain-aware symmetric k-nearest-neighbor graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapgtv.representation.scaling import RobustScaler
from rapgtv.graph.knn import deterministic_knn_union


@dataclass(frozen=True, slots=True)
class PhysicalGraph:
    edge_index: NDArray[np.int64]
    distances: NDArray[np.float64]
    terrain_distances: NDArray[np.float64]
    sigma_s: float
    sigma_b: float
    terrain_affinity: NDArray[np.float64]
    geometry_affinity: NDArray[np.float64]
    terrain_scaler: RobustScaler
    audit_flags: tuple[str, ...]


def _deterministic_knn_union(coords: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    return deterministic_knn_union(coords, k)


def _positive_exponential(exponent: NDArray[np.float64]) -> NDArray[np.float64]:
    floor = np.log(np.finfo(np.float64).tiny)
    return np.exp(np.maximum(exponent, floor))


def build_physical_graph(
    coords_projected: ArrayLike,
    elevation: ArrayLike,
    slope: ArrayLike,
    aspect_radians: ArrayLike,
    curvature: ArrayLike,
    tri: ArrayLike,
    k_s: int,
    *,
    epsilon: float = 1e-8,
) -> PhysicalGraph:
    """Build one topology and both Full-terrain and GeometryOnly affinities."""

    coords = np.asarray(coords_projected, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError("coords_projected must be a finite N by 2 metric array")
    n_nodes = coords.shape[0]
    fields = [np.asarray(x, dtype=np.float64) for x in (elevation, slope, aspect_radians, curvature, tri)]
    if any(x.shape != (n_nodes,) or not np.all(np.isfinite(x)) for x in fields):
        raise ValueError("every terrain field must be finite with shape (N,)")
    terrain = np.column_stack((fields[0], fields[1], np.sin(fields[2]), np.cos(fields[2]), fields[3], fields[4]))
    terrain_scaler = RobustScaler(epsilon).fit(terrain)
    scaled_terrain = terrain_scaler.transform(terrain)
    edges = _deterministic_knn_union(coords, k_s)
    left, right = edges[:, 0], edges[:, 1]
    distances = np.linalg.norm(coords[left] - coords[right], axis=1)
    terrain_distances = np.linalg.norm(scaled_terrain[left] - scaled_terrain[right], axis=1)

    raw_sigma_s = float(np.median(distances))
    raw_sigma_b = float(np.median(terrain_distances))
    sigma_s = max(raw_sigma_s, epsilon)
    sigma_b = max(raw_sigma_b, epsilon)
    flags = []
    if raw_sigma_s <= epsilon:
        flags.append("degenerate_spatial_scale")
    if raw_sigma_b <= epsilon:
        flags.append("degenerate_terrain_scale")
    geometry_exponent = -(distances**2) / (2.0 * sigma_s**2)
    geometry_affinity = _positive_exponential(geometry_exponent)
    terrain_affinity = _positive_exponential(
        geometry_exponent - (terrain_distances**2) / (2.0 * sigma_b**2)
    )
    return PhysicalGraph(
        edges,
        distances,
        terrain_distances,
        sigma_s,
        sigma_b,
        terrain_affinity,
        geometry_affinity,
        terrain_scaler,
        tuple(flags),
    )
