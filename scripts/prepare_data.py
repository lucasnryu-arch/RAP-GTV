from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rapgtv.temporal.split import build_temporal_split


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


def _ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids, str)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and materialize an aligned RAP-GTV site dataset.")
    parser.add_argument("--site", required=True, choices=("xiongba", "offida", "mendatica", "zhouqu_xieliupo"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-points", type=int)
    args = parser.parse_args()

    dataset = _loader(args.site)(data_root=args.data_root, max_points=args.max_points)
    split = build_temporal_split(dataset.times)
    destination = args.output_dir or Path("data") / "processed" / args.site
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "dataset.npz",
        point_ids=dataset.point_ids,
        times=dataset.times,
        displacement=dataset.displacement,
        original_valid_mask=dataset.original_valid_mask,
        coords_projected=dataset.coords_projected,
        elevation=dataset.elevation,
        slope=dataset.slope,
        aspect=dataset.aspect,
        curvature=dataset.curvature,
        tri=dataset.tri,
        displacement_unit=np.asarray(dataset.displacement_unit),
    )
    summary = {
        "site": dataset.site_id,
        "n_points": dataset.n_points,
        "n_epochs": dataset.n_times,
        "point_ids_sha256": _ids_digest(dataset.point_ids),
        "split": {
            "n_train": int(split.train_idx.size),
            "n_validation": int(split.val_idx.size),
            "n_test": int(split.test_idx.size),
        },
        "displacement_unit": dataset.displacement_unit,
        "crs_projected": dataset.crs_projected,
    }
    (destination / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
