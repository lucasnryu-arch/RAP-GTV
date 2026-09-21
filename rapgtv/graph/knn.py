"""Fast deterministic symmetric Euclidean kNN topology shared by science graphs."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def deterministic_knn_union(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n < 2 or not isinstance(k, (int, np.integer)) or k < 1 or k >= n:
        raise ValueError("k must be an integer in [1, N-1]")
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1, workers=1)
    edges: set[tuple[int, int]] = set()
    for source in range(n):
        candidates = np.asarray(indices[source], dtype=np.int64)
        candidate_distances = np.asarray(distances[source], dtype=np.float64)
        keep = candidates != source
        candidates, candidate_distances = candidates[keep], candidate_distances[keep]
        boundary = float(candidate_distances[min(k - 1, candidate_distances.size - 1)])
        tied = np.asarray(tree.query_ball_point(coords[source], boundary + 1e-12), dtype=np.int64)
        tied = tied[tied != source]
        tied_distances = np.linalg.norm(coords[tied] - coords[source], axis=1)
        neighbors = tied[np.lexsort((tied, tied_distances))[:k]]
        for target in neighbors:
            pair = (source, int(target)) if source < target else (int(target), source)
            edges.add(pair)
    if not edges:
        raise RuntimeError("kNN edge set is empty")
    return np.asarray(sorted(edges), dtype=np.int64)
