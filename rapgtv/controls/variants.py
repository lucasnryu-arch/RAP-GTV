"""Mechanism switches that reuse one set of Phase-2A core objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapgtv.graph.weights import AdaptiveEdgeWeights, compute_edge_conductance
from rapgtv.solvers.recovery import RecoveryResult, solve_graph_laplacian, solve_rap_gtv


Regularizer = Literal["graph_tv", "graph_laplacian"]


@dataclass(frozen=True, slots=True)
class RAPVariant:
    use_reliability: bool = True
    use_terrain_affinity: bool = True
    use_optical_gate: bool = True
    regularizer: Regularizer = "graph_tv"


@dataclass(frozen=True, slots=True)
class VariantInputs:
    rho: NDArray[np.float64]
    physical_affinity: NDArray[np.float64]
    optical_gate: NDArray[np.float64]
    edge_weights: AdaptiveEdgeWeights
    regularizer: Regularizer


CANONICAL_VARIANTS: dict[str, RAPVariant] = {
    "Full RAP-GTV": RAPVariant(),
    "NoReliability": RAPVariant(use_reliability=False),
    "GeometryOnly": RAPVariant(use_terrain_affinity=False),
    "NoOptical": RAPVariant(use_optical_gate=False),
    "GraphLaplacian": RAPVariant(regularizer="graph_laplacian"),
}


def prepare_variant_inputs(
    variant: RAPVariant,
    rho: ArrayLike,
    terrain_affinity: ArrayLike,
    geometry_affinity: ArrayLike,
    optical_gate: ArrayLike,
) -> VariantInputs:
    """Apply only the named mechanism switch and recompute the edge budget."""

    fidelity = np.asarray(rho, dtype=np.float64)
    terrain = np.asarray(terrain_affinity, dtype=np.float64)
    geometry = np.asarray(geometry_affinity, dtype=np.float64)
    gate = np.asarray(optical_gate, dtype=np.float64)
    if fidelity.ndim != 1 or not np.all(np.isfinite(fidelity)) or np.any(fidelity <= 0):
        raise ValueError("rho must be a finite positive vector")
    if terrain.ndim != 1 or geometry.shape != terrain.shape or gate.shape != terrain.shape:
        raise ValueError("terrain, geometry, and optical edge arrays must have one shared shape")
    if variant.regularizer not in ("graph_tv", "graph_laplacian"):
        raise ValueError("unknown regularizer")
    selected_rho = fidelity.copy() if variant.use_reliability else np.ones_like(fidelity)
    selected_physical = terrain.copy() if variant.use_terrain_affinity else geometry.copy()
    selected_gate = gate.copy() if variant.use_optical_gate else np.ones_like(gate)
    edge_weights = compute_edge_conductance(selected_physical, selected_gate)
    return VariantInputs(
        selected_rho,
        selected_physical,
        selected_gate,
        edge_weights,
        variant.regularizer,
    )


def solve_variant(
    z_d: ArrayLike,
    edge_index: ArrayLike,
    s_delta: float,
    lambda_: float,
    inputs: VariantInputs,
    *,
    device: str = "cpu",
    max_iters: int | None = None,
    tol: float | None = None,
) -> RecoveryResult:
    """Dispatch one prepared variant to the shared TV or GL solver."""

    common = {
        "z_d": z_d,
        "rho": inputs.rho,
        "edge_index": edge_index,
        "omega": inputs.edge_weights.omega,
        "s_delta": s_delta,
        "lambda_": lambda_,
        "device": device,
    }
    if inputs.regularizer == "graph_tv":
        options = {}
        if max_iters is not None:
            options["max_iters"] = max_iters
        if tol is not None:
            options["tol"] = tol
        return solve_rap_gtv(**common, **options)
    options = {}
    if max_iters is not None:
        options["max_iters"] = max_iters
    if tol is not None:
        options["tol"] = tol
    return solve_graph_laplacian(**common, **options)
