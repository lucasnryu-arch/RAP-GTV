"""Canonical temporal-only Time2Feat-KM and the retained smoke proxy."""

from __future__ import annotations

import numpy as np

from rapgtv.clustering.kmeans import kmeans_fit


REFERENCE_COMMIT = "ad247463d0204f182cb9b8ef2b51dc804b29d5dc"


def _canonical_features(series: np.ndarray, pfa_variance: float) -> tuple[np.ndarray, int, int]:
    """Reference-equivalent univariate Time2Feat extraction and deterministic PFA."""

    try:
        import pandas as pd
        from tsfresh import extract_features
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("canonical Time2Feat requires the 'tsfresh' dependency") from exc

    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Time2Feat requires a finite N by T Train-only series")
    n, t = x.shape
    frame = pd.DataFrame({
        "id": np.repeat(np.arange(n, dtype=np.int64), t),
        "time": np.tile(np.arange(t, dtype=np.int64), n),
        "value": x.reshape(-1),
    })
    extracted = extract_features(
        frame, column_id="id", column_sort="time", disable_progressbar=True, n_jobs=0
    ).sort_index(axis=1)
    raw_count = int(extracted.shape[1])
    values = extracted.to_numpy(dtype=np.float64)
    keep = np.all(np.isfinite(values), axis=0) & (np.ptp(values, axis=0) > 0.0)
    values = values[:, keep]
    if values.shape[1] == 0:
        raise RuntimeError("Time2Feat cleaning removed every feature")

    standardized = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0, ddof=0), 1e-12)
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    variance = singular**2
    q = int(np.searchsorted(np.cumsum(variance) / max(float(variance.sum()), 1e-12), pfa_variance) + 1)
    q = min(q, vt.shape[0], standardized.shape[1])
    loadings = vt[:q].T
    feature_clusters = kmeans_fit(loadings, q, seed=0, n_init=10)
    representatives: list[int] = []
    for cluster in range(q):
        members = np.flatnonzero(feature_clusters.labels == cluster)
        if members.size:
            distances = np.sum((loadings[members] - feature_clusters.centers[cluster]) ** 2, axis=1)
            representatives.append(int(members[int(np.argmin(distances))]))
    selected = values[:, np.asarray(sorted(representatives), dtype=np.int64)]
    selected = (selected - selected.mean(axis=0)) / np.maximum(selected.std(axis=0, ddof=0), 1e-12)
    if not np.all(np.isfinite(selected)):
        raise RuntimeError("Time2Feat produced non-finite selected features")
    return selected, raw_count, int(selected.shape[1])


def build_time2feat_features(train_series: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Build the seed-independent frozen Train-only Time2Feat representation once."""

    if params.get("definition") != "time2feat_tsfresh_pfa90_std_kmeans":
        raise ValueError("formal Time2Feat feature materialization requires the canonical definition")
    features, raw_count, selected_count = _canonical_features(
        train_series, float(params.get("pfa_variance", 0.9))
    )
    return features, {
        "canonical": True,
        "reference_commit": REFERENCE_COMMIT,
        "raw_feature_count": raw_count,
        "selected_feature_count": selected_count,
        "train_only": True,
        "temporal_only": True,
    }


def cluster_time2feat_features(features: np.ndarray, k_star: int, params: dict, seed: int):
    """Apply only the formal seed-dependent final KMeans to frozen Time2Feat features."""

    result = kmeans_fit(features, k_star, seed=seed, n_init=int(params.get("n_init", 10)))
    return result.labels, {"backend": "time2feat-tsfresh-pfa+" + result.backend, "converged": True}


def run_time2feat_km(train_series: np.ndarray, k_star: int, params: dict, seed: int):
    """Run either the retained smoke proxy or frozen canonical Time2Feat-KM."""

    definition = params.get("definition")
    if definition == "smoke_zd_proxy":
        result = kmeans_fit(train_series, k_star, seed=seed, n_init=int(params.get("n_init", 4)))
        return result.labels, {"backend": result.backend, "converged": True, "canonical": False}
    if definition != "time2feat_tsfresh_pfa90_std_kmeans":
        raise ValueError("unknown Time2Feat definition")
    features, feature_audit = build_time2feat_features(train_series, params)
    labels, clustering_audit = cluster_time2feat_features(features, k_star, params, seed)
    return labels, {
        **feature_audit,
        **clustering_audit,
    }
