from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rapgtv.baselines import run_rac_dmvc, run_time2feat_km


TIME2FEAT = {
    "definition": "time2feat_tsfresh_pfa90_std_kmeans",
    "pfa_variance": 0.9,
    "n_init": 10,
}
RAC = {
    "definition": "rac_dmvc_noisy_model_3view_symmetric",
    "hidden_dims": [1024, 1024, 1024],
    "latent_dim": 128,
    "drop_rate": 0.2,
    "noise_ratio": 0.5,
    "sigma": 0.07,
    "contrastive_temperature": 0.5,
    "distill_temperature": 0.5,
    "epochs": 100,
    "warmup_epochs": 20,
    "start_rectify_epoch": 20,
    "batch_size": 1024,
    "momentum": 0.98,
    "base_lr": 0.0005,
    "weight_decay": 0.0,
    "kmeans_n_init": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project-authored comparison-method adapters.")
    parser.add_argument("--method", required=True, choices=("time2feat-km", "rac-dmvc"))
    parser.add_argument("--views-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    with np.load(args.views_npz, allow_pickle=False) as data:
        k_star = int(data["k_star"])
        if args.method == "time2feat-km":
            labels, _ = run_time2feat_km(data["train_series"], k_star, TIME2FEAT, args.seed)
        else:
            labels, _ = run_rac_dmvc(
                (data["z_d"], data["terrain_view"], data["z_o"]),
                k_star,
                RAC,
                args.seed,
                args.device,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)
    print(f"saved {labels.size} aligned labels to {args.output}")


if __name__ == "__main__":
    main()
