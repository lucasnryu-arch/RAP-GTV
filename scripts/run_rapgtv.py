from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rapgtv.clustering import kmeans_fit, select_k
from rapgtv.graph.weights import compute_edge_conductance, compute_jump_scale
from rapgtv.metrics.real import compute_fdd
from rapgtv.metrics.spatial import build_evaluation_graph, compute_fi
from rapgtv.optical import build_optical_representation, load_openeo_composites
from rapgtv.optical.core import compute_optical_gate
from rapgtv.reliability.node import QualityMetadata, compute_node_reliability
from rapgtv.representation.deformation import build_deformation_representation, build_working_displacement
from rapgtv.representation.scaling import RobustScaler
from rapgtv.solvers.recovery import solve_rap_gtv
from rapgtv.temporal.causality import build_construction_context
from rapgtv.temporal.split import build_temporal_split
from rapgtv.terrain.graph import build_physical_graph


def _loader(site: str):
    if site == "xiongba":
        from rapgtv.data.xiongba import load_xiongba

        return load_xiongba
    if site == "offida":
        from rapgtv.data.offida import load_offida

        return load_offida
    if site == "mendatica":
        from rapgtv.data.mendatica import load_mendatica

        return load_mendatica
    if site == "zhouqu_xieliupo":
        from rapgtv.data.zhouqu import load_zhouqu

        return load_zhouqu
    raise ValueError(f"unknown site: {site}")


def _quality_inputs(dataset) -> dict[str, QualityMetadata]:
    accepted = {
        str(item["field"]): str(item["direction"])
        for item in dataset.metadata["quality_metadata_audit"]
        if item["status"] == "ACCEPTED" and item["direction"] is not None
    }
    return {
        name: QualityMetadata(np.asarray(dataset.quality_metadata[name], float), direction)
        for name, direction in accepted.items()
        if name in dataset.quality_metadata
    }


def _terrain_view(dataset) -> np.ndarray:
    raw = np.column_stack(
        (
            dataset.elevation,
            dataset.slope,
            np.sin(np.deg2rad(dataset.aspect)),
            np.cos(np.deg2rad(dataset.aspect)),
            dataset.curvature,
            dataset.tri,
        )
    )
    return RobustScaler().fit_transform(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAP-GTV for one prepared real site.")
    parser.add_argument("--site", required=True, choices=("xiongba", "offida", "mendatica", "zhouqu_xieliupo"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--optical-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-points", type=int)
    args = parser.parse_args()

    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / f"{args.site}.json").read_text(encoding="utf-8"))
    dataset = _loader(args.site)(data_root=args.data_root, max_points=args.max_points)
    split = build_temporal_split(dataset.times)
    context = build_construction_context(split, "train")
    deformation = build_deformation_representation(dataset, context)
    selected_k = select_k(deformation.z_d, seed=0, n_init=20)
    if args.max_points is None and selected_k.k_star != int(config["k_star"]):
        raise RuntimeError(
            f"Train-derived K*={selected_k.k_star} does not match the site setting {config['k_star']}"
        )
    k_star = selected_k.k_star if args.max_points is not None else int(config["k_star"])
    composites = load_openeo_composites(args.optical_provenance, dataset.coords_projected)
    optical = build_optical_representation(composites, context)
    reliability = compute_node_reliability(dataset, context, _quality_inputs(dataset))
    graph = build_physical_graph(
        dataset.coords_projected,
        dataset.elevation,
        dataset.slope,
        np.deg2rad(dataset.aspect),
        dataset.curvature,
        dataset.tri,
        int(config["k_s"]),
    )
    gate = compute_optical_gate(optical.z_o, optical.q_o, graph.edge_index)
    weights = compute_edge_conductance(graph.terrain_affinity, gate.gate)
    scale = compute_jump_scale(deformation.z_d, graph.edge_index)
    recovered = solve_rap_gtv(
        deformation.z_d,
        reliability.rho,
        graph.edge_index,
        weights.omega,
        scale,
        float(config["lambda"]),
        device=args.device,
        max_iters=int(config["solver"]["max_iters"]),
        tol=float(config["solver"]["tolerance"]),
    )
    zoning = kmeans_fit(
        recovered.latent,
        k_star,
        seed=args.seed,
        n_init=int(config["clustering"]["n_init"]),
        max_iter=int(config["clustering"]["max_iter"]),
        device=args.device,
    )
    evaluation_graph = build_evaluation_graph(dataset.coords_projected, int(config["fi_k_eval"]))
    metrics = {
        "site": args.site,
        "seed": args.seed,
        "n_points": dataset.n_points,
        "k_star": k_star,
        "FDD": compute_fdd(dataset.displacement, zoning.labels, split.test_idx),
        "FDD_unit": str(dataset.metadata.get("fdd_unit", dataset.displacement_unit)),
        "FI": compute_fi(zoning.labels, evaluation_graph),
        "converged": recovered.converged,
        "iterations": recovered.iterations,
        "objective": recovered.final_objective,
        "runtime_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"labels_seed{args.seed}.npy", zoning.labels)
    np.savez_compressed(
        args.output_dir / "views.npz",
        train_series=build_working_displacement(dataset, context),
        z_d=deformation.z_d,
        terrain_view=_terrain_view(dataset),
        z_o=optical.z_o,
        k_star=np.asarray(k_star),
    )
    (args.output_dir / f"metrics_seed{args.seed}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
