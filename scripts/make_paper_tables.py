from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "site",
    "method",
    "FDD_mean",
    "FDD_sample_sd",
    "FDD_unit",
    "FI_mean",
    "FI_sample_sd",
    "rank_FDD",
    "rank_FI",
    "seed_count",
]


def build_table(reference: Path) -> pd.DataFrame:
    table = pd.read_csv(reference)
    if table.shape[0] != 16 or set(table["site"]) != {
        "Xiongba",
        "Offida",
        "Mendatica",
        "Zhouqu-Xieliupo",
    }:
        raise ValueError("Table 2 reference must contain four methods at each of four sites")
    calculated_fdd = table.groupby("site")["FDD_mean"].rank(method="min").astype(int)
    calculated_fi = table.groupby("site")["FI_mean"].rank(method="min").astype(int)
    if not calculated_fdd.equals(table["rank_FDD"].astype(int)):
        raise ValueError("stored FDD ranks do not match the means")
    if not calculated_fi.equals(table["rank_FI"].astype(int)):
        raise ValueError("stored FI ranks do not match the means")
    return table[COLUMNS].copy()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Regenerate the manuscript Table 2 CSV.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "paper_results" / "table2_real_benchmark.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper_results" / "generated_table2.csv",
    )
    args = parser.parse_args()
    generated = build_table(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(args.output, index=False)
    print(f"wrote {len(generated)} rows to {args.output}")


if __name__ == "__main__":
    main()
