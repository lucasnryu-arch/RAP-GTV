import numpy as np

from rapgtv.graph.weights import compute_edge_conductance
from rapgtv.optical.core import compute_optical_gate
from rapgtv.terrain.graph import build_physical_graph


def test_graph_construction_is_deterministic() -> None:
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 2], [2, 2]], dtype=float)
    n = len(coords)
    args = (
        coords,
        np.linspace(100, 140, n),
        np.linspace(0.1, 0.5, n),
        np.linspace(0, np.pi, n),
        np.linspace(-1, 1, n),
        np.linspace(2, 5, n),
        2,
    )
    first = build_physical_graph(*args)
    second = build_physical_graph(*args)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    assert np.all((first.terrain_affinity > 0) & (first.terrain_affinity <= 1))


def test_zero_optical_reliability_is_neutral() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)
    features = np.array([[0.0], [10.0], [20.0], [30.0]])
    gate = compute_optical_gate(features, np.zeros(4), edges)
    np.testing.assert_array_equal(gate.gate, np.ones(edges.shape[0]))
    weights = compute_edge_conductance(np.array([0.2, 0.5, 0.9]), gate.gate)
    assert np.isclose(np.mean(weights.omega), 1.0)
