"""Deterministic exact NumPy Lloyd KMeans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    labels: NDArray[np.int64]
    centers: NDArray[np.float64]
    inertia: float
    iterations: int
    seed: int
    n_init: int
    backend: str = "numpy_exact_lloyd"


def _squared_distances(values: NDArray[np.float64], centers: NDArray[np.float64]) -> NDArray[np.float64]:
    squared = (
        np.sum(values * values, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (values @ centers.T)
    )
    return np.maximum(squared, 0.0)


def _kmeans_plus_plus(
    values: NDArray[np.float64],
    n_clusters: int,
    rng: np.random.Generator,
    epsilon: float,
) -> NDArray[np.float64]:
    n_samples = values.shape[0]
    selected: list[int] = [int(rng.integers(n_samples))]
    centers = [values[selected[0]].copy()]
    closest = np.sum((values - centers[0]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        total = float(np.sum(closest))
        if total <= epsilon:
            unselected = np.setdiff1d(np.arange(n_samples), np.asarray(selected), assume_unique=False)
            index = int(unselected[0])
        else:
            index = int(rng.choice(n_samples, p=closest / total))
        selected.append(index)
        centers.append(values[index].copy())
        closest = np.minimum(closest, np.sum((values - values[index]) ** 2, axis=1))
    return np.asarray(centers)


def _single_lloyd(
    values: NDArray[np.float64],
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int,
    epsilon: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float, int]:
    centers = _kmeans_plus_plus(values, n_clusters, rng, epsilon)
    previous_labels: NDArray[np.int64] | None = None
    node_ids = np.arange(values.shape[0])
    for iteration in range(1, max_iter + 1):
        squared = _squared_distances(values, centers)
        labels = np.argmin(squared, axis=1).astype(np.int64)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        assigned = squared[node_ids, labels]
        new_centers = np.empty_like(centers)
        empty_clusters = []
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if members.size:
                new_centers[cluster] = np.mean(members, axis=0)
            else:
                empty_clusters.append(cluster)
        if empty_clusters:
            farthest = np.lexsort((node_ids, -assigned))
            used: set[int] = set()
            cursor = 0
            for cluster in empty_clusters:
                while int(farthest[cursor]) in used:
                    cursor += 1
                point = int(farthest[cursor])
                used.add(point)
                new_centers[cluster] = values[point]
                cursor += 1
        centers = new_centers
        previous_labels = labels
    squared = _squared_distances(values, centers)
    labels = np.argmin(squared, axis=1).astype(np.int64)
    inertia = float(np.sum(squared[node_ids, labels]))
    return labels, centers, inertia, iteration


def _single_lloyd_torch_cuda(
    values: NDArray[np.float64],
    values_t,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int,
    epsilon: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float, int]:
    import torch

    centers_np = _kmeans_plus_plus(values, n_clusters, rng, epsilon)
    centers = torch.as_tensor(centers_np, dtype=torch.float64, device=values_t.device)
    previous_labels = None
    node_ids = torch.arange(values.shape[0], device=values_t.device)
    for iteration in range(1, max_iter + 1):
        squared = torch.clamp(
            torch.sum(values_t * values_t, dim=1)[:, None]
            + torch.sum(centers * centers, dim=1)[None, :]
            - 2.0 * (values_t @ centers.T),
            min=0.0,
        )
        labels = torch.argmin(squared, dim=1)
        if previous_labels is not None and bool(torch.equal(labels, previous_labels)):
            break
        assigned = squared[node_ids, labels]
        new_centers = torch.empty_like(centers)
        empty_clusters = []
        for cluster in range(n_clusters):
            members = values_t[labels == cluster]
            if members.numel():
                new_centers[cluster] = torch.mean(members, dim=0)
            else:
                empty_clusters.append(cluster)
        if empty_clusters:
            farthest = torch.argsort(assigned, descending=True, stable=True).detach().cpu().tolist()
            used: set[int] = set()
            cursor = 0
            for cluster in empty_clusters:
                while int(farthest[cursor]) in used:
                    cursor += 1
                point = int(farthest[cursor])
                used.add(point)
                new_centers[cluster] = values_t[point]
                cursor += 1
        centers = new_centers
        previous_labels = labels
    squared = torch.clamp(
        torch.sum(values_t * values_t, dim=1)[:, None]
        + torch.sum(centers * centers, dim=1)[None, :]
        - 2.0 * (values_t @ centers.T),
        min=0.0,
    )
    labels = torch.argmin(squared, dim=1)
    inertia = float(torch.sum(squared[node_ids, labels]).item())
    return labels.detach().cpu().numpy(), centers.detach().cpu().numpy(), inertia, iteration


def kmeans_fit(
    values: ArrayLike,
    n_clusters: int,
    *,
    seed: int = 0,
    n_init: int = 20,
    max_iter: int = 300,
    epsilon: float = 1e-12,
    device: str = "cpu",
) -> ClusteringResult:
    """Fit exact full-data KMeans with deterministic k-means++ restarts."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("KMeans values must be a finite non-empty N by p array")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1 or n_clusters > array.shape[0]:
        raise ValueError("n_clusters must be an integer in [1, N]")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive integers")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    values_t = None
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        values_t = torch.as_tensor(array, dtype=torch.float64, device="cuda")

    best: tuple[NDArray[np.int64], NDArray[np.float64], float, int] | None = None
    for restart in range(int(n_init)):
        rng = np.random.default_rng(int(seed) + 104729 * (restart + 1))
        candidate = (
            _single_lloyd(array, int(n_clusters), rng, int(max_iter), epsilon)
            if device == "cpu"
            else _single_lloyd_torch_cuda(array, values_t, int(n_clusters), rng, int(max_iter), epsilon)
        )
        if best is None or candidate[2] < best[2] - epsilon:
            best = candidate
    assert best is not None
    backend = "numpy_exact_lloyd" if device == "cpu" else "torch_cuda_float64_exact_lloyd"
    return ClusteringResult(best[0], best[1], best[2], best[3], int(seed), int(n_init), backend)
