"""Frozen controlled-ground-truth metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def nrmse_u(estimated: ArrayLike, truth: ArrayLike, *, epsilon: float = 1e-12) -> float:
    """Return Frobenius latent recovery error normalized by truth magnitude."""

    estimate = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimated and truth latent fields must have one shared N by p shape")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(target)):
        raise ValueError("latent fields must be finite")
    return float(np.linalg.norm(estimate - target) / max(float(np.linalg.norm(target)), epsilon))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), sorted_scores.size - 1]
    precision = true_positive[group_ends] / (true_positive[group_ends] + false_positive[group_ends])
    recall = true_positive[group_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def boundary_auprc(
    estimated_latent: ArrayLike,
    truth_labels: ArrayLike,
    edge_index: ArrayLike,
    s_delta: float,
) -> float:
    """Compute boundary average precision from continuous normalized jumps."""

    latent = np.asarray(estimated_latent, dtype=np.float64)
    labels = np.asarray(truth_labels)
    edges = np.asarray(edge_index, dtype=np.int64)
    if latent.ndim != 2 or not np.all(np.isfinite(latent)):
        raise ValueError("estimated_latent must be a finite N by p array")
    if labels.ndim != 1 or labels.shape[0] != latent.shape[0]:
        raise ValueError("truth_labels must have shape (N,)")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        raise ValueError("edge_index must be a non-empty E by 2 array")
    if np.any(edges < 0) or np.any(edges >= latent.shape[0]):
        raise ValueError("edge_index contains invalid endpoints")
    if not np.isfinite(s_delta) or s_delta <= 0:
        raise ValueError("s_delta must be finite and positive")
    boundary = labels[edges[:, 0]] != labels[edges[:, 1]]
    scores = np.linalg.norm(latent[edges[:, 0]] - latent[edges[:, 1]], axis=1) / s_delta
    return _average_precision(boundary, scores)


def _comb2(value: int) -> int:
    return math.comb(int(value), 2) if value >= 2 else 0


def adjusted_rand_index(labels_a: ArrayLike, labels_b: ArrayLike) -> float:
    """Standard adjusted Rand index with deterministic input validation."""

    left = np.asarray(labels_a)
    right = np.asarray(labels_b)
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError("ARI inputs must be matching non-empty label vectors")
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    contingency = np.zeros((left_inverse.max() + 1, right_inverse.max() + 1), dtype=np.int64)
    np.add.at(contingency, (left_inverse, right_inverse), 1)
    sum_cells = sum(_comb2(int(value)) for value in contingency.ravel())
    sum_rows = sum(_comb2(int(value)) for value in contingency.sum(axis=1))
    sum_columns = sum(_comb2(int(value)) for value in contingency.sum(axis=0))
    total_pairs = _comb2(left.size)
    if total_pairs == 0:
        return 1.0
    expected = sum_rows * sum_columns / total_pairs
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    if abs(denominator) <= 1e-15:
        return 1.0 if abs(sum_cells - maximum) <= 1e-15 else 0.0
    return float((sum_cells - expected) / denominator)
