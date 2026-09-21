from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rapgtv.metrics.real import compute_fdd
from rapgtv.metrics.spatial import build_evaluation_graph, compute_fi
from rapgtv.temporal.split import build_temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate aligned regime labels with FDD and FI.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = np.load(args.labels, allow_pickle=False)
    with np.load(args.dataset_npz, allow_pickle=False) as data:
        displacement = data["displacement"]
        times = data["times"]
        coords = data["coords_projected"]
        unit = str(data["displacement_unit"])
    if labels.shape != (displacement.shape[0],) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be an integer vector aligned to every dataset point")
    split = build_temporal_split(times)
    graph = build_evaluation_graph(coords, 8)
    result = {
        "FDD": compute_fdd(displacement, labels, split.test_idx),
        "FDD_unit": unit,
        "FI": compute_fi(labels, graph),
        "k_eval": 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
