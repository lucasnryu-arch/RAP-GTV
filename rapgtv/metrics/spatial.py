"""Coordinate-only evaluation graph and Fragmentation Index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rapgtv.graph.knn import deterministic_knn_union


@dataclass(frozen=True, slots=True)
class EvaluationGraph:
    edge_index: NDArray[np.int64]
    n_nodes: int
    k_eval: int


def build_evaluation_graph(coords_projected: ArrayLike, k_eval: int) -> EvaluationGraph:
    """Build a deterministic symmetric coordinate-only kNN union."""

    coords = np.asarray(coords_projected, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError("coords_projected must be a finite N by 2 array")
    n_nodes = coords.shape[0]
    if not isinstance(k_eval, (int, np.integer)) or k_eval < 1 or k_eval >= n_nodes:
        raise ValueError("k_eval must be an integer in [1, N-1]")
    return EvaluationGraph(deterministic_knn_union(coords, int(k_eval)), n_nodes, int(k_eval))


def compute_fi(labels: ArrayLike, evaluation_graph: EvaluationGraph) -> float:
    """Return the mean number of induced connected components per cluster."""

    clusters = np.asarray(labels)
    if clusters.ndim != 1 or clusters.shape[0] != evaluation_graph.n_nodes or clusters.size == 0:
        raise ValueError("FI labels must have shape (evaluation_graph.n_nodes,)")
    parent = np.arange(clusters.size, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    for left, right in evaluation_graph.edge_index:
        if clusters[left] == clusters[right]:
            union(int(left), int(right))
    component_counts = []
    for label in np.unique(clusters):
        members = np.flatnonzero(clusters == label)
        component_counts.append(len({find(int(member)) for member in members}))
    return float(np.mean(component_counts))
