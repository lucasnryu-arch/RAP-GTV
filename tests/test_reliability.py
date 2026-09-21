import numpy as np

from conftest import make_site_dataset
from rapgtv.reliability.node import QualityMetadata, compute_metadata_reliability, compute_node_reliability
from rapgtv.temporal.causality import build_construction_context
from rapgtv.temporal.split import build_temporal_split


def test_reliability_mapping_and_mean_one_normalization() -> None:
    high = np.linspace(0.0, 1.0, 20)
    low = high[::-1]
    mapped = compute_metadata_reliability(
        {
            "high": QualityMetadata(high, "higher_is_better"),
            "low": QualityMetadata(low, "lower_is_better"),
        },
        20,
    )
    assert np.all(np.diff(mapped) >= 0)
    site = make_site_dataset()
    context = build_construction_context(build_temporal_split(site.times), "train")
    result = compute_node_reliability(
        site,
        context,
        {"coherence": QualityMetadata(site.quality_metadata["coherence"], "higher_is_better")},
    )
    assert np.isclose(np.mean(result.rho), 1.0, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(result.rho))
