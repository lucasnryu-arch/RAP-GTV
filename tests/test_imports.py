def test_public_imports() -> None:
    import rapgtv
    from rapgtv.clustering import kmeans_fit, select_k
    from rapgtv.graph import compute_edge_conductance, compute_jump_scale
    from rapgtv.metrics import build_evaluation_graph, compute_fdd, compute_fi
    from rapgtv.optical import compute_optical_gate
    from rapgtv.reliability import compute_node_reliability
    from rapgtv.solvers import solve_rap_gtv
    from rapgtv.terrain import build_physical_graph

    assert rapgtv.__version__ == "0.9.0"
    assert all(
        callable(item)
        for item in (
            kmeans_fit,
            select_k,
            compute_edge_conductance,
            compute_jump_scale,
            build_evaluation_graph,
            compute_fdd,
            compute_fi,
            compute_optical_gate,
            compute_node_reliability,
            solve_rap_gtv,
            build_physical_graph,
        )
    )
