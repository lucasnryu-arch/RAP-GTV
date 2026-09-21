"""CPU-only synthetic demonstration of the complete RAP-GTV core pipeline."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rapgtv.clustering import kmeans_fit, select_k
from rapgtv.data.schema import SiteDataset
from rapgtv.graph.weights import compute_edge_conductance, compute_jump_scale
from rapgtv.optical.core import compute_optical_gate
from rapgtv.reliability.node import QualityMetadata, compute_node_reliability
from rapgtv.representation.deformation import build_deformation_representation
from rapgtv.solvers.recovery import solve_rap_gtv
from rapgtv.temporal.causality import build_construction_context
from rapgtv.temporal.split import build_temporal_split
from rapgtv.terrain.graph import build_physical_graph


def synthetic_site(side: int = 6, epochs: int = 36) -> SiteDataset:
    rng = np.random.default_rng(7)
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.column_stack((cols.ravel(), rows.ravel())).astype(float) * 20.0
    n = coords.shape[0]
    times = np.datetime64("2020-01-01") + np.arange(epochs) * np.timedelta64(12, "D")
    regime = (cols.ravel() >= side // 2).astype(int)
    trend = np.where(regime[:, None] == 0, 0.08, 0.24) * np.arange(epochs)[None, :]
    seasonal = 0.12 * np.sin(np.arange(epochs)[None, :] / 4.0 + rows.ravel()[:, None] / 5.0)
    displacement = trend + seasonal + rng.normal(scale=0.025, size=(n, epochs))
    valid = np.ones((n, epochs), dtype=bool)
    valid[::11, 3] = False
    displacement[~valid] = np.nan
    ids = np.asarray([f"synthetic-{i:03d}" for i in range(n)])
    quality = {"coherence": np.linspace(0.65, 0.95, n)}
    aligned = {
        "displacement",
        "original_valid_mask",
        "coords_projected",
        "elevation",
        "slope",
        "aspect",
        "curvature",
        "tri",
        "quality_metadata.coherence",
    }
    return SiteDataset(
        site_id="synthetic",
        point_ids=ids,
        times=times,
        displacement=displacement,
        original_valid_mask=valid,
        coords_projected=coords,
        coords_lonlat=None,
        quality_metadata=quality,
        elevation=1000.0 + 8.0 * rows.ravel() + 4.0 * regime,
        slope=10.0 + rows.ravel(),
        aspect=np.full(n, 90.0),
        curvature=np.linspace(-0.1, 0.1, n),
        tri=1.0 + 0.1 * rows.ravel(),
        optical_source=None,
        crs_projected="synthetic-metre-grid",
        crs_geographic=None,
        displacement_unit="synthetic",
        projected_coordinate_unit="metre",
        alignment_ids={name: ids.copy() for name in aligned},
        metadata={"source": "deterministic synthetic example"},
    )


def run_example() -> dict[str, object]:
    started = perf_counter()
    site = synthetic_site()
    split = build_temporal_split(site.times)
    context = build_construction_context(split, "train")
    representation = build_deformation_representation(site, context)
    reliability = compute_node_reliability(
        site,
        context,
        {"coherence": QualityMetadata(site.quality_metadata["coherence"], "higher_is_better")},
    )
    graph = build_physical_graph(
        site.coords_projected,
        site.elevation,
        site.slope,
        np.deg2rad(site.aspect),
        site.curvature,
        site.tri,
        4,
    )
    optical = np.column_stack((site.coords_projected[:, 0] / 100.0, site.coords_projected[:, 1] / 100.0))
    optical_reliability = np.ones(site.n_points)
    optical_reliability[0] = 0.0
    gate = compute_optical_gate(optical, optical_reliability, graph.edge_index)
    weights = compute_edge_conductance(graph.terrain_affinity, gate.gate)
    jump_scale = compute_jump_scale(representation.z_d, graph.edge_index)
    recovery = solve_rap_gtv(
        representation.z_d,
        reliability.rho,
        graph.edge_index,
        weights.omega,
        jump_scale,
        0.08,
        device="cpu",
        max_iters=5000,
        tol=1e-7,
    )
    selection = select_k(representation.z_d, seed=7, n_init=5)
    zoning = kmeans_fit(recovery.latent, selection.k_star, seed=7, n_init=10)
    return {
        "nodes": site.n_points,
        "regimes": int(np.unique(zoning.labels).size),
        "selected_k": selection.k_star,
        "converged": recovery.converged,
        "runtime_seconds": perf_counter() - started,
        "rho_mean": float(np.mean(reliability.rho)),
        "finite_recovery": bool(np.all(np.isfinite(recovery.latent))),
    }


def main() -> None:
    result = run_example()
    print(
        "nodes={nodes} regimes={regimes} selected_k={selected_k} "
        "convergence={converged} runtime={runtime_seconds:.3f}s".format(**result)
    )


if __name__ == "__main__":
    main()
