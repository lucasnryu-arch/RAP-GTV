from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rapgtv.clustering.kmeans import kmeans_fit
from rapgtv.controlled import ControlledCondition, generate_controlled_realization
from rapgtv.controls.variants import CANONICAL_VARIANTS, prepare_variant_inputs, solve_variant
from rapgtv.graph.weights import compute_jump_scale
from rapgtv.metrics.controlled import adjusted_rand_index, boundary_auprc, nrmse_u
from rapgtv.optical.core import compute_optical_gate
from rapgtv.terrain.graph import build_physical_graph


def _cases(config: dict) -> list[tuple[str, str, float | str, ControlledCondition, tuple[str, ...]]]:
    result = []
    for h in config["H_levels"]:
        result.append(("A", "H", h, ControlledCondition(h, 1, 1, 2.0, 0.0), ("Full RAP-GTV", "NoReliability")))
    for value in config["eta_T_levels"]:
        result.append(("B1", "eta_T", value, ControlledCondition(1, value, 1, 2.0, 0.0), ("Full RAP-GTV", "GeometryOnly")))
    for value in config["eta_O_levels"]:
        result.append(("B2", "eta_O", value, ControlledCondition(1, 0, value, 2.0, 0.8), ("Full RAP-GTV", "NoOptical")))
    for jump in config["J_levels"]:
        result.append(("C", "J", jump, ControlledCondition(1, 1, 1, jump, 0.0), ("Full RAP-GTV", "GraphLaplacian")))
    result.append(("D", "target", "combined", ControlledCondition(8, 1, 1, 2.0, 0.8), tuple(CANONICAL_VARIANTS)))
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the controlled RAP-GTV comparisons.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", type=Path, default=root / "configs" / "controlled.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = config["seeds"][:2] if args.quick else config["seeds"]
    rows = []
    for block, factor_name, factor_value, condition, variants in _cases(config):
        for seed in seeds:
            realization = generate_controlled_realization(condition, seed)
            graph = build_physical_graph(
                realization.coordinates,
                realization.elevation,
                realization.slope,
                realization.aspect_radians,
                realization.curvature,
                realization.tri,
                int(config["k_s"]),
            )
            gate = compute_optical_gate(realization.z_o, realization.q_o, graph.edge_index)
            scale = compute_jump_scale(realization.z_d, graph.edge_index)
            for variant_name in variants:
                inputs = prepare_variant_inputs(
                    CANONICAL_VARIANTS[variant_name],
                    realization.rho,
                    graph.terrain_affinity,
                    graph.geometry_affinity,
                    gate.gate,
                )
                recovered = solve_variant(
                    realization.z_d,
                    graph.edge_index,
                    scale,
                    float(config["lambda"]),
                    inputs,
                    device=config["solver"]["device"],
                    max_iters=int(config["solver"]["max_iters"]),
                    tol=float(config["solver"]["tolerance"]),
                )
                labels = kmeans_fit(recovered.latent, 3, seed=seed, n_init=10).labels
                values = {
                    "NRMSE_U": nrmse_u(recovered.latent, realization.u_star),
                    "Boundary_AUPRC": boundary_auprc(recovered.latent, realization.true_labels, realization.evaluation_edges, scale),
                    "ARI": adjusted_rand_index(labels, realization.true_labels),
                }
                for metric, value in values.items():
                    rows.append(
                        {
                            "subexperiment": block,
                            "factor_name": factor_name,
                            "factor_value": factor_value,
                            "variant": variant_name,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                            "converged": recovered.converged,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["subexperiment", "factor_name", "factor_value", "variant", "metric"], dropna=False)["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "controlled_raw.csv", index=False)
    summary.to_csv(args.output_dir / "controlled_summary.csv", index=False)
    print(f"wrote {len(raw)} metric rows to {args.output_dir}")


if __name__ == "__main__":
    main()
