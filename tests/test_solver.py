import numpy as np

from rapgtv.solvers.recovery import solve_rap_gtv


def test_graph_tv_small_problem_is_finite_and_converged() -> None:
    result = solve_rap_gtv(
        np.array([[0.0], [4.0]]),
        np.ones(2),
        np.array([[0, 1]]),
        np.ones(1),
        2.0,
        0.5,
        max_iters=10000,
        tol=1e-10,
    )
    np.testing.assert_allclose(result.latent[:, 0], [0.5, 3.5], rtol=0, atol=2e-6)
    assert result.converged
    assert np.all(np.isfinite(result.latent))
