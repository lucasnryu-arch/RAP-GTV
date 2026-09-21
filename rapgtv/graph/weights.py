"""Step-4 edge conductance and graph jump scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class AdaptiveEdgeWeights:
    conductance: NDArray[np.float64]
    omega: NDArray[np.float64]
    used_uniform_fallback: bool
    audit_flags: tuple[str, ...]


def compute_edge_conductance(
    physical_affinity: ArrayLike,
    optical_gate: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> AdaptiveEdgeWeights:
    """Compute ``c=P*gO`` and normalize its edge budget to mean one."""

    physical = np.asarray(physical_affinity, dtype=np.float64)
    gate = np.asarray(optical_gate, dtype=np.float64)
    if physical.ndim != 1 or physical.size == 0 or gate.shape != physical.shape:
        raise ValueError("physical_affinity and optical_gate must be matching non-empty vectors")
    if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(gate)):
        raise ValueError("edge inputs must be finite")
    if np.any(physical < 0) or np.any((gate < 0) | (gate > 1)):
        raise ValueError("physical affinity must be nonnegative and gate must lie in [0, 1]")
    conductance = physical * gate
    total = float(np.sum(conductance))
    if total <= epsilon:
        omega = np.ones_like(conductance)
        return AdaptiveEdgeWeights(conductance, omega, True, ("uniform_conductance_fallback",))
    omega = conductance.size * conductance / total
    if not np.isclose(np.mean(omega), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("edge regularization budget normalization failed")
    return AdaptiveEdgeWeights(conductance, omega, False, ())


def compute_jump_scale(
    z_d: ArrayLike,
    edge_index: ArrayLike,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Return the unique data-adaptive Step-4 local latent jump scale."""

    latent = np.asarray(z_d, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if latent.ndim != 2 or not np.all(np.isfinite(latent)):
        raise ValueError("z_d must be a finite two-dimensional array")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        raise ValueError("edge_index must be a non-empty E by 2 array")
    if np.any(edges < 0) or np.any(edges >= latent.shape[0]):
        raise ValueError("edge_index references a node outside z_d")
    jumps = np.linalg.norm(latent[edges[:, 0]] - latent[edges[:, 1]], axis=1)
    return float(np.median(jumps) + epsilon)
