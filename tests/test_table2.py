from pathlib import Path

import pandas as pd

from scripts.make_paper_tables import build_table


def test_table2_reference_consistency(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    reference_path = root / "paper_results" / "table2_real_benchmark.csv"
    reference = pd.read_csv(reference_path)
    generated = build_table(reference_path)
    pd.testing.assert_frame_equal(generated, reference, check_exact=False, rtol=0, atol=1e-14)
    assert generated.shape == (16, 10)
    rap = generated[generated["method"] == "RAP-GTV"]
    assert rap["rank_FI"].tolist() == [1, 1, 1, 1]
    assert rap["rank_FDD"].tolist() == [2, 1, 1, 1]
