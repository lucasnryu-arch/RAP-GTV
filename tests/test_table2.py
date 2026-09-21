from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_table2_reference_consistency(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    reference_path = root / "paper_results" / "table2_real_benchmark.csv"
    generated_path = tmp_path / "generated_table2.csv"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_paper_tables.py"),
            "--reference",
            str(reference_path),
            "--output",
            str(generated_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    reference = pd.read_csv(reference_path)
    generated = pd.read_csv(generated_path)
    pd.testing.assert_frame_equal(generated, reference, check_exact=False, rtol=0, atol=1e-14)
    assert generated.shape == (16, 10)
    rap = generated[generated["method"] == "RAP-GTV"]
    assert rap["rank_FI"].tolist() == [1, 1, 1, 1]
    assert rap["rank_FDD"].tolist() == [2, 1, 1, 1]
