"""Frozen deformation-only Train K-star elbow rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kmeans import kmeans_fit


@dataclass(frozen=True, slots=True)
class KSelectionResult:
    k_star: int
    candidates: NDArray[np.int64]
    inertia: NDArray[np.float64]
    normalized_inertia: NDArray[np.float64]
    line: NDArray[np.float64]
    elbow_score: NDArray[np.float64]
    degenerate_inertia_span: bool
    seed: int
    n_init: int


def select_k(
    z_d_train: ArrayLike,
    *,
    seed: int = 0,
    n_init: int = 20,
    epsilon: float = 1e-12,
) -> KSelectionResult:
    """Select shared K-star from only ``Z_D_train`` using Step-4 equations."""

    values = np.asarray(z_d_train, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 10 or not np.all(np.isfinite(values)):
        raise ValueError("the frozen K=2,...,10 rule requires finite Z_D_train with N >= 10")
    candidates = np.arange(2, 11, dtype=np.int64)
    inertia = np.asarray(
        [kmeans_fit(values, int(k), seed=seed, n_init=n_init).inertia for k in candidates],
        dtype=np.float64,
    )
    span = float(inertia[0] - inertia[-1])
    degenerate = abs(span) <= epsilon
    if degenerate:
        normalized = np.zeros_like(inertia)
    else:
        normalized = (inertia - inertia[-1]) / (span + epsilon)
    line = (10.0 - candidates) / 8.0
    score = line - normalized
    eligible = (candidates >= 3) & (candidates <= 9)
    # np.argmax over ascending candidates freezes a smallest-K tie break.
    k_star = int(candidates[eligible][np.argmax(score[eligible])])
    return KSelectionResult(
        k_star,
        candidates,
        inertia,
        normalized,
        line,
        score,
        degenerate,
        int(seed),
        int(n_init),
    )
