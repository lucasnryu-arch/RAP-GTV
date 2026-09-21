from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("Time2Feat-KM", "K-SC", "RAC-DMVC", "RAP-GTV")
COLORS = ("#4C78A8", "#F58518", "#E45756", "#54A24B")


def controlled_figure(source: Path, destination: Path) -> None:
    import matplotlib.pyplot as plt

    data = pd.read_csv(source)
    panel = data[
        (data["subexperiment"] == "A")
        & (data["metric"] == "NRMSE_U")
        & data["variant"].isin(["Full RAP-GTV", "NoReliability"])
    ].copy()
    figure, axis = plt.subplots(figsize=(5.0, 3.2))
    for color, (name, rows) in zip(("#54A24B", "#B279A2"), panel.groupby("variant")):
        rows = rows.sort_values("factor_value")
        axis.errorbar(rows["factor_value"], rows["mean"], yerr=rows["std"], marker="o", label=name, color=color)
    axis.set(xlabel="Observation-noise contrast H", ylabel="NRMSE")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)


def real_benchmark_figure(source: Path, destination: Path) -> None:
    import matplotlib.pyplot as plt

    data = pd.read_csv(source)
    sites = list(dict.fromkeys(data["site"]))
    figure, axes = plt.subplots(2, 4, figsize=(11.0, 5.4))
    for column, site in enumerate(sites):
        rows = data[data["site"] == site].set_index("method").loc[list(METHODS)]
        positions = np.arange(len(METHODS))
        axes[0, column].bar(positions, rows["FDD_mean"], yerr=rows["FDD_sample_sd"], color=COLORS)
        axes[1, column].bar(positions, rows["FI_mean"], yerr=rows["FI_sample_sd"], color=COLORS)
        axes[0, column].set_title(site)
        axes[0, column].set_ylabel("FDD" if column == 0 else "")
        axes[1, column].set_ylabel("FI" if column == 0 else "")
        axes[1, column].set_xticks(positions, ("T2F", "K-SC", "RAC", "RAP"), rotation=35, ha="right")
        axes[0, column].set_xticks([])
        axes[0, column].spines[["top", "right"]].set_visible(False)
        axes[1, column].spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate result-based manuscript figures.")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "paper_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    controlled_figure(
        root / "paper_results" / "controlled_summary.csv",
        args.output_dir / "figure3_controlled_summary.pdf",
    )
    real_benchmark_figure(
        root / "paper_results" / "table2_real_benchmark.csv",
        args.output_dir / "figure6_quantitative_comparison.pdf",
    )
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
