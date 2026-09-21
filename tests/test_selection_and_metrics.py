import numpy as np

from rapgtv.clustering.selection import select_k
from rapgtv.metrics.real import compute_fdd
from rapgtv.metrics.spatial import build_evaluation_graph, compute_fi


def test_kstar_selection_is_deterministic() -> None:
    rng = np.random.default_rng(12)
    values = np.vstack((rng.normal(-2, 0.1, (12, 2)), rng.normal(0, 0.1, (12, 2)), rng.normal(2, 0.1, (12, 2))))
    first = select_k(values, seed=3, n_init=4)
    second = select_k(values, seed=3, n_init=4)
    assert first.k_star == second.k_star
    np.testing.assert_array_equal(first.inertia, second.inertia)


def test_fdd_and_coordinate_only_fi_k8() -> None:
    displacement = np.array(
        [[0, 1, 2, 3], [5, 6, 7, 8], [0, -1, -2, -3], [8, 7, 6, 5]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])
    assert compute_fdd(displacement, labels, np.array([1, 2, 3])) == 0.0
    coords = np.column_stack((np.arange(12, dtype=float), np.zeros(12)))
    graph = build_evaluation_graph(coords, 8)
    value = compute_fi(np.repeat([0, 1], 6), graph)
    assert np.isfinite(value)
    assert graph.edge_index.shape[1] == 2
