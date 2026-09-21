"""Numerical recovery for the exact frozen TV and matched GL objectives."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FORMAL_RELATIVE_PRIMAL_DUAL_GAP_TOL = 1e-6
FORMAL_DUAL_FEASIBILITY_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    latent: NDArray[np.float64]
    converged: bool
    iterations: int
    final_objective: float
    final_relative_change: float
    solver_name: str
    device: str
    diagnostics: dict[str, Any]
    dual: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class RAPGTVCertificate:
    """Exact Fenchel certificate on the frozen, publication-scale objective."""

    primal_objective: float
    dual_objective: float
    primal_dual_gap: float
    relative_primal_dual_gap: float
    dual_feasibility_violation: float
    stationarity_residual: float


def _validated_inputs(
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    latent = np.asarray(z_d, dtype=np.float64)
    fidelity = np.asarray(rho, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    weights = np.asarray(omega, dtype=np.float64)
    if latent.ndim != 2 or latent.shape[0] == 0 or latent.shape[1] == 0 or not np.all(np.isfinite(latent)):
        raise ValueError("z_d must be a finite non-empty N by p array")
    if fidelity.shape != (latent.shape[0],) or not np.all(np.isfinite(fidelity)) or np.any(fidelity <= 0):
        raise ValueError("rho must be finite, strictly positive, and have shape (N,)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_index must have shape (E, 2)")
    if np.any(edges < 0) or np.any(edges >= latent.shape[0]) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("edge_index contains an invalid endpoint or self-edge")
    if weights.shape != (edges.shape[0],) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("omega must be finite, nonnegative, and have shape (E,)")
    if not np.isfinite(s_delta) or s_delta <= 0:
        raise ValueError("s_delta must be finite and positive")
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative")
    return latent, fidelity, edges, weights


def _fidelity_value(u: NDArray[np.float64], z: NDArray[np.float64], rho: NDArray[np.float64]) -> float:
    return float(np.sum(rho[:, None] * (u - z) ** 2) / (2.0 * z.shape[0]))


def compute_rap_gtv_objective(
    u: ArrayLike,
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
) -> float:
    """Evaluate the frozen RAP-GTV objective exactly once for solver and tests."""

    z, fidelity, edges, weights = _validated_inputs(z_d, rho, edge_index, omega, s_delta, lambda_)
    candidate = np.asarray(u, dtype=np.float64)
    if candidate.shape != z.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("u must be finite and have the same shape as z_d")
    if edges.shape[0] == 0 or lambda_ == 0:
        return _fidelity_value(candidate, z, fidelity)
    jumps = np.linalg.norm(candidate[edges[:, 0]] - candidate[edges[:, 1]], axis=1)
    regularizer = lambda_ * float(np.dot(weights, jumps)) / (edges.shape[0] * s_delta)
    return _fidelity_value(candidate, z, fidelity) + regularizer


def compute_rap_gtv_certificate(
    u: ArrayLike,
    dual: ArrayLike,
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
    *,
    dual_scaling: str = "solver",
) -> RAPGTVCertificate:
    """Evaluate the exact primal/dual certificate for frozen RAP-GTV.

    The solver minimizes ``N * P(U)`` and stores its dual variable ``Y`` with
    ``||Y_e|| <= lambda*N*omega_e/(|E|*s_delta)``.  With
    ``K U = U[source] - U[target]``, the publication-scale dual is

    ``D(Y) = (<K.T Y, Z> - .5*sum_i ||(K.T Y)_i||^2/rho_i) / N``.

    Passing ``dual_scaling='objective'`` instead accepts ``P=Y/N`` directly.
    A finite primal-dual gap is a valid upper bound only when the reported
    feasibility violation is zero up to floating-point roundoff.
    """

    z, fidelity, edges, weights = _validated_inputs(z_d, rho, edge_index, omega, s_delta, lambda_)
    candidate = np.asarray(u, dtype=np.float64)
    multiplier = np.asarray(dual, dtype=np.float64)
    if candidate.shape != z.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("u must be finite and have the same shape as z_d")
    if multiplier.shape != (edges.shape[0], z.shape[1]) or not np.all(np.isfinite(multiplier)):
        raise ValueError("dual must be finite and have shape (E, p)")
    if dual_scaling not in ("solver", "objective"):
        raise ValueError("dual_scaling must be 'solver' or 'objective'")

    n_nodes, n_edges = z.shape[0], edges.shape[0]
    solver_dual = multiplier if dual_scaling == "solver" else n_nodes * multiplier
    if n_edges:
        radius = lambda_ * n_nodes * weights / (n_edges * s_delta)
        norms = np.linalg.norm(solver_dual, axis=1)
        absolute_violation = float(np.max(np.maximum(norms - radius, 0.0), initial=0.0))
        feasibility_scale = max(float(np.max(radius, initial=0.0)), np.finfo(np.float64).tiny)
        feasibility = absolute_violation / feasibility_scale
        divergence = np.zeros_like(candidate)
        np.add.at(divergence, edges[:, 0], solver_dual)
        np.add.at(divergence, edges[:, 1], -solver_dual)
    else:
        feasibility = 0.0
        divergence = np.zeros_like(candidate)

    primal = compute_rap_gtv_objective(candidate, z, fidelity, edges, weights, s_delta, lambda_)
    dual_value = float(
        (np.sum(divergence * z) - 0.5 * np.sum((divergence * divergence) / fidelity[:, None]))
        / n_nodes
    )
    gap = float(primal - dual_value)
    gap_scale = max(abs(primal), abs(dual_value), np.finfo(np.float64).tiny)
    relative_gap = max(gap, 0.0) / gap_scale
    fidelity_gradient = fidelity[:, None] * (candidate - z)
    stationarity = float(
        np.linalg.norm(fidelity_gradient + divergence)
        / max(np.linalg.norm(fidelity_gradient), np.linalg.norm(divergence), np.finfo(np.float64).tiny)
    )
    return RAPGTVCertificate(
        primal,
        dual_value,
        gap,
        relative_gap,
        feasibility,
        stationarity,
    )


def compute_graph_laplacian_objective(
    u: ArrayLike,
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
) -> float:
    """Evaluate the matched ``psi_GL(r)=r^2/2`` objective."""

    z, fidelity, edges, weights = _validated_inputs(z_d, rho, edge_index, omega, s_delta, lambda_)
    candidate = np.asarray(u, dtype=np.float64)
    if candidate.shape != z.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("u must be finite and have the same shape as z_d")
    if edges.shape[0] == 0 or lambda_ == 0:
        return _fidelity_value(candidate, z, fidelity)
    squared_jumps = np.sum((candidate[edges[:, 0]] - candidate[edges[:, 1]]) ** 2, axis=1)
    regularizer = lambda_ * float(np.dot(weights, squared_jumps)) / (
        2.0 * edges.shape[0] * s_delta**2
    )
    return _fidelity_value(candidate, z, fidelity) + regularizer


def _shortcut_result(
    z: NDArray[np.float64],
    objective: Callable[[NDArray[np.float64]], float],
    solver_name: str,
    device: str,
    reason: str,
) -> RecoveryResult:
    value = objective(z)
    return RecoveryResult(
        z.copy(), True, 0, value, 0.0, solver_name, device, {"shortcut": reason, "initial_objective": value}
    )


def _tv_numpy(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    s_delta: float,
    lambda_: float,
    max_iters: int,
    tol: float,
    check_every: int,
    consecutive_tol_checks: int,
    stopping_criterion: str,
    algorithm: str,
    initial_latent: NDArray[np.float64] | None,
    initial_dual: NDArray[np.float64] | None,
    initial_extrapolated: NDArray[np.float64] | None,
    iteration_offset: int,
    checkpoint_every: int | None,
    checkpoint_callback: Callable[[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]], None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float, dict[str, Any]]:
    n_nodes = z.shape[0]
    n_edges = edges.shape[0]
    source, target = edges[:, 0], edges[:, 1]
    degree = np.bincount(edges.reshape(-1), minlength=n_nodes)
    dmax = max(int(np.max(degree)), 1)
    if algorithm == "diagonal_pdhg":
        # Standard diagonal CP preconditioning (alpha=1) after the exact
        # variable change X_i=sqrt(rho_i) U_i.  This balances both incidence
        # endpoints and is the empirically retained admissible member.
        sqrt_rho = np.sqrt(rho)
        primal_step = 0.99 / (np.maximum(degree, 1) * sqrt_rho)
        dual_step = 0.99 / (1.0 / sqrt_rho[source] + 1.0 / sqrt_rho[target])
    else:
        scalar_step = 0.99 / np.sqrt(2.0 * dmax)
        primal_step = np.full(n_nodes, scalar_step, dtype=np.float64)
        dual_step = np.full(n_edges, scalar_step, dtype=np.float64)
    lambda_solver = lambda_ * n_nodes / (n_edges * s_delta)
    radius = lambda_solver * omega[:, None]
    denominator = 1.0 + primal_step[:, None] * rho[:, None]
    rho_z = primal_step[:, None] * rho[:, None] * z

    u = z.copy() if initial_latent is None else initial_latent.copy()
    u_bar = u.copy() if initial_extrapolated is None else initial_extrapolated.copy()
    dual = np.zeros((n_edges, z.shape[1]), dtype=np.float64) if initial_dual is None else initial_dual.copy()
    norm = np.linalg.norm(dual, axis=1, keepdims=True)
    dual *= np.minimum(1.0, radius / np.maximum(norm, 1e-30))
    good_checks = 0
    converged = False
    relative_change = float("inf")
    history: list[dict[str, float | int]] = []
    for local_iteration in range(1, max_iters - iteration_offset + 1):
        iteration = iteration_offset + local_iteration
        dual += dual_step[:, None] * (u_bar[source] - u_bar[target])
        norm = np.linalg.norm(dual, axis=1, keepdims=True)
        dual *= np.minimum(1.0, radius / np.maximum(norm, 1e-30))

        divergence = np.zeros_like(u)
        np.add.at(divergence, source, dual)
        np.add.at(divergence, target, -dual)
        candidate = (u - primal_step[:, None] * divergence + rho_z) / denominator
        u_bar = 2.0 * candidate - u

        if iteration == 1 or iteration % check_every == 0 or iteration == max_iters:
            relative_change = float(np.linalg.norm(candidate - u) / max(np.linalg.norm(u), 1e-30))
            certificate = compute_rap_gtv_certificate(
                candidate, dual, z, rho, edges, omega, s_delta, lambda_
            )
            stopping_value = (
                certificate.relative_primal_dual_gap
                if stopping_criterion == "relative_primal_dual_gap"
                else relative_change
            )
            certificate_ok = certificate.dual_feasibility_violation <= FORMAL_DUAL_FEASIBILITY_TOL
            good_checks = good_checks + 1 if stopping_value <= tol and certificate_ok else 0
            history.append(
                {
                    "iteration": iteration,
                    "relative_change": relative_change,
                    "primal_objective": certificate.primal_objective,
                    "dual_objective": certificate.dual_objective,
                    "primal_dual_gap": certificate.primal_dual_gap,
                    "relative_primal_dual_gap": certificate.relative_primal_dual_gap,
                    "dual_feasibility_violation": certificate.dual_feasibility_violation,
                    "stationarity_residual": certificate.stationarity_residual,
                    "consecutive_good_checks": good_checks,
                }
            )
            if checkpoint_callback is not None and iteration % checkpoint_every == 0:
                checkpoint_callback(
                    iteration, candidate.copy(), dual.copy(), u_bar.copy(), dict(history[-1])
                )
            if good_checks >= consecutive_tol_checks:
                u = candidate
                converged = True
                break
        u = candidate
    diagnostics = {
        "algorithm": algorithm,
        "step_tau_min": float(np.min(primal_step)),
        "step_tau_median": float(np.median(primal_step)),
        "step_tau_max": float(np.max(primal_step)),
        "step_sigma_min": float(np.min(dual_step)),
        "step_sigma_max": float(np.max(dual_step)),
        "maximum_degree": dmax,
        "lambda_solver": lambda_solver,
        "stopping_criterion": stopping_criterion,
        "history": history,
    }
    return u, dual, converged, iteration, relative_change, diagnostics


def _tv_torch_cuda(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    s_delta: float,
    lambda_: float,
    max_iters: int,
    tol: float,
    check_every: int,
    consecutive_tol_checks: int,
    stopping_criterion: str,
    algorithm: str,
    initial_latent: NDArray[np.float64] | None,
    initial_dual: NDArray[np.float64] | None,
    initial_extrapolated: NDArray[np.float64] | None,
    iteration_offset: int,
    checkpoint_every: int | None,
    checkpoint_callback: Callable[[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]], None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dev = torch.device("cuda")
    zt = torch.as_tensor(z, dtype=torch.float64, device=dev)
    rhot = torch.as_tensor(rho, dtype=torch.float64, device=dev)
    edge_t = torch.as_tensor(edges, dtype=torch.int64, device=dev)
    wt = torch.as_tensor(omega, dtype=torch.float64, device=dev)
    source, target = edge_t[:, 0], edge_t[:, 1]
    degree = torch.bincount(edge_t.reshape(-1), minlength=z.shape[0])
    dmax = max(int(degree.max().item()), 1)
    if algorithm == "diagonal_pdhg":
        sqrt_rho = torch.sqrt(rhot)
        primal_step = 0.99 / (torch.clamp(degree, min=1).to(torch.float64) * sqrt_rho)
        dual_step = 0.99 / (1.0 / sqrt_rho[source] + 1.0 / sqrt_rho[target])
    else:
        scalar_step = 0.99 / np.sqrt(2.0 * dmax)
        primal_step = torch.full((z.shape[0],), scalar_step, dtype=torch.float64, device=dev)
        dual_step = torch.full((edges.shape[0],), scalar_step, dtype=torch.float64, device=dev)
    lambda_solver = lambda_ * z.shape[0] / (edges.shape[0] * s_delta)
    radius = (lambda_solver * wt)[:, None]
    denominator = 1.0 + primal_step[:, None] * rhot[:, None]
    rho_z = primal_step[:, None] * rhot[:, None] * zt
    u = zt.clone() if initial_latent is None else torch.as_tensor(initial_latent, dtype=torch.float64, device=dev).clone()
    u_bar = (
        u.clone() if initial_extrapolated is None
        else torch.as_tensor(initial_extrapolated, dtype=torch.float64, device=dev).clone()
    )
    dual = (
        torch.zeros((edges.shape[0], z.shape[1]), dtype=torch.float64, device=dev)
        if initial_dual is None
        else torch.as_tensor(initial_dual, dtype=torch.float64, device=dev).clone()
    )
    norm = torch.linalg.vector_norm(dual, dim=1, keepdim=True)
    dual.mul_(torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-30)))
    good_checks = 0
    converged = False
    relative_change = float("inf")
    history = []
    with torch.inference_mode():
        for local_iteration in range(1, max_iters - iteration_offset + 1):
            iteration = iteration_offset + local_iteration
            dual.add_(dual_step[:, None] * (u_bar[source] - u_bar[target]))
            norm = torch.linalg.vector_norm(dual, dim=1, keepdim=True)
            dual.mul_(torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-30)))
            divergence = torch.zeros_like(u)
            divergence.index_add_(0, source, dual)
            divergence.index_add_(0, target, -dual)
            candidate = (u - primal_step[:, None] * divergence + rho_z) / denominator
            u_bar = 2.0 * candidate - u
            if iteration == 1 or iteration % check_every == 0 or iteration == max_iters:
                relative_change = float(
                    (torch.linalg.vector_norm(candidate - u) / torch.clamp(torch.linalg.vector_norm(u), min=1e-30)).item()
                )
                jumps = candidate[source] - candidate[target]
                primal = (
                    0.5 * torch.sum(rhot[:, None] * (candidate - zt) ** 2) / z.shape[0]
                    + lambda_ * torch.sum(wt * torch.linalg.vector_norm(jumps, dim=1))
                    / (edges.shape[0] * s_delta)
                )
                dual_objective = (
                    torch.sum(divergence * zt)
                    - 0.5 * torch.sum((divergence * divergence) / rhot[:, None])
                ) / z.shape[0]
                gap = primal - dual_objective
                relative_gap = torch.clamp(gap, min=0.0) / torch.clamp(
                    torch.maximum(torch.abs(primal), torch.abs(dual_objective)), min=torch.finfo(torch.float64).tiny
                )
                absolute_violation = torch.max(
                    torch.clamp(torch.linalg.vector_norm(dual, dim=1) - radius[:, 0], min=0.0)
                )
                feasibility = absolute_violation / torch.clamp(torch.max(radius), min=torch.finfo(torch.float64).tiny)
                fidelity_gradient = rhot[:, None] * (candidate - zt)
                stationarity = torch.linalg.vector_norm(fidelity_gradient + divergence) / torch.clamp(
                    torch.maximum(
                        torch.linalg.vector_norm(fidelity_gradient), torch.linalg.vector_norm(divergence)
                    ),
                    min=torch.finfo(torch.float64).tiny,
                )
                stopping_value = float(
                    relative_gap.item() if stopping_criterion == "relative_primal_dual_gap" else relative_change
                )
                certificate_ok = float(feasibility.item()) <= FORMAL_DUAL_FEASIBILITY_TOL
                good_checks = good_checks + 1 if stopping_value <= tol and certificate_ok else 0
                history.append(
                    {
                        "iteration": iteration,
                        "relative_change": relative_change,
                        "primal_objective": float(primal.item()),
                        "dual_objective": float(dual_objective.item()),
                        "primal_dual_gap": float(gap.item()),
                        "relative_primal_dual_gap": float(relative_gap.item()),
                        "dual_feasibility_violation": float(feasibility.item()),
                        "stationarity_residual": float(stationarity.item()),
                        "consecutive_good_checks": good_checks,
                    }
                )
                if checkpoint_callback is not None and iteration % checkpoint_every == 0:
                    checkpoint_callback(
                        iteration,
                        candidate.detach().cpu().numpy(),
                        dual.detach().cpu().numpy(),
                        u_bar.detach().cpu().numpy(),
                        dict(history[-1]),
                    )
                if good_checks >= consecutive_tol_checks:
                    u = candidate
                    converged = True
                    break
            u = candidate
    return (
        u.detach().cpu().numpy(),
        dual.detach().cpu().numpy(),
        converged,
        iteration,
        relative_change,
        {
            "algorithm": algorithm,
            "step_tau_min": float(torch.min(primal_step).item()),
            "step_tau_median": float(torch.median(primal_step).item()),
            "step_tau_max": float(torch.max(primal_step).item()),
            "step_sigma_min": float(torch.min(dual_step).item()),
            "step_sigma_max": float(torch.max(dual_step).item()),
            "maximum_degree": dmax,
            "lambda_solver": lambda_solver,
            "stopping_criterion": stopping_criterion,
            "history": history,
        },
    )


def _dual_fista_numpy(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    s_delta: float,
    lambda_: float,
    max_iters: int,
    tol: float,
    check_every: int,
    consecutive_tol_checks: int,
    stopping_criterion: str,
    algorithm: str,
    initial_latent: NDArray[np.float64] | None,
    initial_dual: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float, dict[str, Any]]:
    del initial_latent, algorithm
    n_nodes, n_edges = z.shape[0], edges.shape[0]
    source, target = edges[:, 0], edges[:, 1]
    degree = np.bincount(edges.reshape(-1), minlength=n_nodes).astype(np.float64)
    radius = (lambda_ * n_nodes * omega / (n_edges * s_delta))[:, None]
    step = 0.99 / (degree[source] / rho[source] + degree[target] / rho[target])
    dual = np.zeros((n_edges, z.shape[1]), dtype=np.float64) if initial_dual is None else initial_dual.copy()
    norm = np.linalg.norm(dual, axis=1, keepdims=True)
    dual *= np.minimum(1.0, radius / np.maximum(norm, 1e-30))
    extrapolated = dual.copy()
    momentum = 1.0
    previous_u = z.copy()
    relative_change = float("inf")
    converged = False
    good_checks = 0
    history: list[dict[str, float | int]] = []
    for iteration in range(1, max_iters + 1):
        divergence_q = np.zeros_like(z)
        np.add.at(divergence_q, source, extrapolated)
        np.add.at(divergence_q, target, -extrapolated)
        u_q = z - divergence_q / rho[:, None]
        candidate_dual = extrapolated + step[:, None] * (u_q[source] - u_q[target])
        norm = np.linalg.norm(candidate_dual, axis=1, keepdims=True)
        candidate_dual *= np.minimum(1.0, radius / np.maximum(norm, 1e-30))
        divergence = np.zeros_like(z)
        np.add.at(divergence, source, candidate_dual)
        np.add.at(divergence, target, -candidate_dual)
        candidate_u = z - divergence / rho[:, None]
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = candidate_dual + ((momentum - 1.0) / next_momentum) * (candidate_dual - dual)
        dual = candidate_dual
        momentum = next_momentum
        if iteration == 1 or iteration % check_every == 0 or iteration == max_iters:
            relative_change = float(
                np.linalg.norm(candidate_u - previous_u) / max(np.linalg.norm(previous_u), np.finfo(np.float64).tiny)
            )
            certificate = compute_rap_gtv_certificate(
                candidate_u, dual, z, rho, edges, omega, s_delta, lambda_
            )
            stopping_value = certificate.relative_primal_dual_gap if stopping_criterion == "relative_primal_dual_gap" else relative_change
            certificate_ok = certificate.dual_feasibility_violation <= FORMAL_DUAL_FEASIBILITY_TOL
            good_checks = good_checks + 1 if stopping_value <= tol and certificate_ok else 0
            history.append({
                "iteration": iteration, "relative_change": relative_change,
                "primal_objective": certificate.primal_objective,
                "dual_objective": certificate.dual_objective,
                "primal_dual_gap": certificate.primal_dual_gap,
                "relative_primal_dual_gap": certificate.relative_primal_dual_gap,
                "dual_feasibility_violation": certificate.dual_feasibility_violation,
                "stationarity_residual": certificate.stationarity_residual,
                "consecutive_good_checks": good_checks,
            })
            previous_u = candidate_u.copy()
            if good_checks >= consecutive_tol_checks:
                converged = True
                break
    diagnostics = {
        "algorithm": "dual_fista", "dual_step_min": float(np.min(step)),
        "dual_step_median": float(np.median(step)), "dual_step_max": float(np.max(step)),
        "lambda_solver": lambda_ * n_nodes / (n_edges * s_delta),
        "stopping_criterion": stopping_criterion, "history": history,
    }
    return candidate_u, dual, converged, iteration, relative_change, diagnostics


def _dual_fista_torch_cuda(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    s_delta: float,
    lambda_: float,
    max_iters: int,
    tol: float,
    check_every: int,
    consecutive_tol_checks: int,
    stopping_criterion: str,
    algorithm: str,
    initial_latent: NDArray[np.float64] | None,
    initial_dual: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float, dict[str, Any]]:
    del initial_latent, algorithm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dev = torch.device("cuda")
    zt = torch.as_tensor(z, dtype=torch.float64, device=dev)
    rhot = torch.as_tensor(rho, dtype=torch.float64, device=dev)
    edge_t = torch.as_tensor(edges, dtype=torch.int64, device=dev)
    wt = torch.as_tensor(omega, dtype=torch.float64, device=dev)
    source, target = edge_t[:, 0], edge_t[:, 1]
    degree = torch.bincount(edge_t.reshape(-1), minlength=z.shape[0]).to(torch.float64)
    radius = (lambda_ * z.shape[0] * wt / (edges.shape[0] * s_delta))[:, None]
    step = 0.99 / (degree[source] / rhot[source] + degree[target] / rhot[target])
    dual = (
        torch.zeros((edges.shape[0], z.shape[1]), dtype=torch.float64, device=dev)
        if initial_dual is None else torch.as_tensor(initial_dual, dtype=torch.float64, device=dev).clone()
    )
    norm = torch.linalg.vector_norm(dual, dim=1, keepdim=True)
    dual.mul_(torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-30)))
    extrapolated = dual.clone()
    momentum = 1.0
    previous_u = zt.clone()
    relative_change = float("inf")
    converged = False
    good_checks = 0
    history = []
    with torch.inference_mode():
        for iteration in range(1, max_iters + 1):
            divergence_q = torch.zeros_like(zt)
            divergence_q.index_add_(0, source, extrapolated)
            divergence_q.index_add_(0, target, -extrapolated)
            u_q = zt - divergence_q / rhot[:, None]
            candidate_dual = extrapolated + step[:, None] * (u_q[source] - u_q[target])
            norm = torch.linalg.vector_norm(candidate_dual, dim=1, keepdim=True)
            candidate_dual.mul_(torch.minimum(torch.ones_like(norm), radius / torch.clamp(norm, min=1e-30)))
            divergence = torch.zeros_like(zt)
            divergence.index_add_(0, source, candidate_dual)
            divergence.index_add_(0, target, -candidate_dual)
            candidate_u = zt - divergence / rhot[:, None]
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = candidate_dual + ((momentum - 1.0) / next_momentum) * (candidate_dual - dual)
            dual = candidate_dual
            momentum = next_momentum
            if iteration == 1 or iteration % check_every == 0 or iteration == max_iters:
                relative_change = float(
                    (torch.linalg.vector_norm(candidate_u - previous_u) / torch.clamp(torch.linalg.vector_norm(previous_u), min=1e-30)).item()
                )
                jumps = candidate_u[source] - candidate_u[target]
                primal = (
                    0.5 * torch.sum(rhot[:, None] * (candidate_u - zt) ** 2) / z.shape[0]
                    + lambda_ * torch.sum(wt * torch.linalg.vector_norm(jumps, dim=1)) / (edges.shape[0] * s_delta)
                )
                dual_objective = (
                    torch.sum(divergence * zt) - 0.5 * torch.sum((divergence * divergence) / rhot[:, None])
                ) / z.shape[0]
                gap = primal - dual_objective
                relative_gap = torch.clamp(gap, min=0.0) / torch.clamp(
                    torch.maximum(torch.abs(primal), torch.abs(dual_objective)), min=torch.finfo(torch.float64).tiny
                )
                feasibility = torch.max(torch.clamp(torch.linalg.vector_norm(dual, dim=1) - radius[:, 0], min=0.0)) / torch.clamp(torch.max(radius), min=torch.finfo(torch.float64).tiny)
                fidelity_gradient = rhot[:, None] * (candidate_u - zt)
                stationarity = torch.linalg.vector_norm(fidelity_gradient + divergence) / torch.clamp(
                    torch.maximum(torch.linalg.vector_norm(fidelity_gradient), torch.linalg.vector_norm(divergence)), min=torch.finfo(torch.float64).tiny
                )
                stopping_value = float(relative_gap.item()) if stopping_criterion == "relative_primal_dual_gap" else relative_change
                good_checks = (
                    good_checks + 1
                    if stopping_value <= tol
                    and float(feasibility.item()) <= FORMAL_DUAL_FEASIBILITY_TOL
                    else 0
                )
                history.append({
                    "iteration": iteration, "relative_change": relative_change,
                    "primal_objective": float(primal.item()), "dual_objective": float(dual_objective.item()),
                    "primal_dual_gap": float(gap.item()), "relative_primal_dual_gap": float(relative_gap.item()),
                    "dual_feasibility_violation": float(feasibility.item()),
                    "stationarity_residual": float(stationarity.item()), "consecutive_good_checks": good_checks,
                })
                previous_u = candidate_u.clone()
                if good_checks >= consecutive_tol_checks:
                    converged = True
                    break
    diagnostics = {
        "algorithm": "dual_fista", "dual_step_min": float(torch.min(step).item()),
        "dual_step_median": float(torch.median(step).item()), "dual_step_max": float(torch.max(step).item()),
        "lambda_solver": lambda_ * z.shape[0] / (edges.shape[0] * s_delta),
        "stopping_criterion": stopping_criterion, "history": history,
    }
    return (
        candidate_u.detach().cpu().numpy(), dual.detach().cpu().numpy(), converged,
        iteration, relative_change, diagnostics,
    )


def solve_rap_gtv(
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
    *,
    device: str = "cpu",
    max_iters: int = 5000,
    tol: float = 1e-8,
    check_every: int = 100,
    consecutive_tol_checks: int = 1,
    stopping_criterion: str = "relative_primal_dual_gap",
    algorithm: str = "diagonal_pdhg",
    warm_start: tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    iteration_offset: int = 0,
    checkpoint_every: int | None = None,
    checkpoint_callback: Callable[[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]], None] | None = None,
) -> RecoveryResult:
    """Solve the canonical weighted vector Graph-TV objective."""

    z, fidelity, edges, weights = _validated_inputs(z_d, rho, edge_index, omega, s_delta, lambda_)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if max_iters < 1 or tol <= 0 or check_every < 1 or consecutive_tol_checks < 1:
        raise ValueError("solver iteration and tolerance settings must be positive")
    if iteration_offset < 0 or iteration_offset >= max_iters:
        raise ValueError("iteration_offset must be in [0, max_iters)")
    if checkpoint_callback is not None:
        if checkpoint_every is None or checkpoint_every < 1 or checkpoint_every % check_every:
            raise ValueError("checkpoint_every must be a positive multiple of check_every")
        if algorithm == "dual_fista":
            raise ValueError("exact intra-candidate checkpoints are implemented for PDHG only")
    if stopping_criterion not in ("relative_primal_dual_gap", "iterate_change"):
        raise ValueError("unknown stopping criterion")
    if algorithm not in ("diagonal_pdhg", "scalar_pdhg", "dual_fista"):
        raise ValueError("algorithm must be 'diagonal_pdhg', 'scalar_pdhg', or 'dual_fista'")
    initial_latent = initial_dual = initial_extrapolated = None
    if warm_start is not None:
        if len(warm_start) not in (2, 3):
            raise ValueError("warm_start must contain latent/dual and optional extrapolated latent")
        initial_latent = np.asarray(warm_start[0], dtype=np.float64)
        initial_dual = np.asarray(warm_start[1], dtype=np.float64)
        if initial_latent.shape != z.shape or initial_dual.shape != (edges.shape[0], z.shape[1]):
            raise ValueError("warm_start must contain latent (N,p) and dual (E,p) arrays")
        if not np.all(np.isfinite(initial_latent)) or not np.all(np.isfinite(initial_dual)):
            raise ValueError("warm_start arrays must be finite")
        if len(warm_start) == 3:
            initial_extrapolated = np.asarray(warm_start[2], dtype=np.float64)
            if initial_extrapolated.shape != z.shape or not np.all(np.isfinite(initial_extrapolated)):
                raise ValueError("warm_start extrapolated latent must be finite with shape (N,p)")
    objective = lambda u: compute_rap_gtv_objective(u, z, fidelity, edges, weights, s_delta, lambda_)
    if edges.shape[0] == 0:
        return _shortcut_result(z, objective, "chambolle_pock_vector_gtv", device, "no_edges")
    if lambda_ == 0:
        return _shortcut_result(z, objective, "chambolle_pock_vector_gtv", device, "zero_lambda")

    initial_objective = objective(z)
    started = perf_counter()
    if algorithm == "dual_fista":
        backend = _dual_fista_numpy if device == "cpu" else _dual_fista_torch_cuda
    else:
        backend = _tv_numpy if device == "cpu" else _tv_torch_cuda
    backend_args = (
        z, fidelity, edges, weights, s_delta, lambda_, max_iters, tol, check_every,
        consecutive_tol_checks, stopping_criterion, algorithm, initial_latent, initial_dual,
    )
    if algorithm == "dual_fista":
        latent, dual, converged, iterations, relative_change, diagnostics = backend(*backend_args)
    else:
        latent, dual, converged, iterations, relative_change, diagnostics = backend(
            *backend_args, initial_extrapolated, iteration_offset, checkpoint_every, checkpoint_callback
        )
    final_objective = objective(latent)
    diagnostics.update(
        {
            "initial_objective": initial_objective,
            "objective_change": final_objective - initial_objective,
            "elapsed_seconds": perf_counter() - started,
            "n_nodes": z.shape[0],
            "n_edges": edges.shape[0],
            "warm_start_used": warm_start is not None,
            "iteration_offset": iteration_offset,
            "intra_candidate_checkpointing": checkpoint_callback is not None,
        }
    )
    if diagnostics.get("history"):
        diagnostics["final_certificate"] = {
            key: diagnostics["history"][-1][key]
            for key in (
                "primal_objective",
                "dual_objective",
                "primal_dual_gap",
                "relative_primal_dual_gap",
                "dual_feasibility_violation",
                "stationarity_residual",
            )
        }
    return RecoveryResult(
        latent,
        converged,
        iterations,
        final_objective,
        relative_change,
        {
            "diagonal_pdhg": "diagonally_preconditioned_pdhg_vector_gtv",
            "scalar_pdhg": "scalar_pdhg_vector_gtv",
            "dual_fista": "diagonally_preconditioned_dual_fista_vector_gtv",
        }[algorithm],
        device,
        diagnostics,
        dual,
    )


def _laplacian_apply_numpy(
    values: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    coefficient: float,
) -> NDArray[np.float64]:
    result = rho[:, None] * values
    difference = omega[:, None] * (values[edges[:, 0]] - values[edges[:, 1]])
    np.add.at(result, edges[:, 0], coefficient * difference)
    np.add.at(result, edges[:, 1], -coefficient * difference)
    return result


def _gl_numpy(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    coefficient: float,
    max_iters: int,
    tol: float,
) -> tuple[NDArray[np.float64], bool, int, float, dict[str, Any]]:
    right_hand_side = rho[:, None] * z
    solution = z.copy()
    residual = right_hand_side - _laplacian_apply_numpy(solution, rho, edges, omega, coefficient)
    direction = residual.copy()
    squared_residual = float(np.sum(residual * residual))
    initial_squared_residual = max(squared_residual, 1e-30)
    relative_residual = np.sqrt(squared_residual / initial_squared_residual)
    if squared_residual <= 1e-30:
        return solution, True, 0, 0.0, {"linear_system_coefficient": coefficient}
    converged = False
    for iteration in range(1, max_iters + 1):
        applied = _laplacian_apply_numpy(direction, rho, edges, omega, coefficient)
        denominator = float(np.sum(direction * applied))
        if denominator <= 0:
            raise RuntimeError("Graph-Laplacian CG lost positive definiteness")
        alpha = squared_residual / denominator
        solution += alpha * direction
        residual -= alpha * applied
        next_squared_residual = float(np.sum(residual * residual))
        relative_residual = float(np.sqrt(next_squared_residual / initial_squared_residual))
        if relative_residual < tol:
            converged = True
            squared_residual = next_squared_residual
            break
        direction = residual + (next_squared_residual / squared_residual) * direction
        squared_residual = next_squared_residual
    return solution, converged, iteration, relative_residual, {"linear_system_coefficient": coefficient}


def _gl_torch_cuda(
    z: NDArray[np.float64],
    rho: NDArray[np.float64],
    edges: NDArray[np.int64],
    omega: NDArray[np.float64],
    coefficient: float,
    max_iters: int,
    tol: float,
) -> tuple[NDArray[np.float64], bool, int, float, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dev = torch.device("cuda")
    zt = torch.as_tensor(z, dtype=torch.float64, device=dev)
    rhot = torch.as_tensor(rho, dtype=torch.float64, device=dev)
    edge_t = torch.as_tensor(edges, dtype=torch.int64, device=dev)
    wt = torch.as_tensor(omega, dtype=torch.float64, device=dev)
    source, target = edge_t[:, 0], edge_t[:, 1]

    def apply(values):
        result = rhot[:, None] * values
        difference = wt[:, None] * (values[source] - values[target])
        result.index_add_(0, source, coefficient * difference)
        result.index_add_(0, target, -coefficient * difference)
        return result

    with torch.inference_mode():
        rhs = rhot[:, None] * zt
        solution = zt.clone()
        residual = rhs - apply(solution)
        direction = residual.clone()
        squared_residual = torch.sum(residual * residual)
        initial = torch.clamp(squared_residual.clone(), min=1e-30)
        if float(squared_residual.item()) <= 1e-30:
            return z.copy(), True, 0, 0.0, {"linear_system_coefficient": coefficient}
        converged = False
        relative_residual = float("inf")
        for iteration in range(1, max_iters + 1):
            applied = apply(direction)
            denominator = torch.sum(direction * applied)
            if float(denominator.item()) <= 0:
                raise RuntimeError("Graph-Laplacian CG lost positive definiteness")
            alpha = squared_residual / denominator
            solution.add_(alpha * direction)
            residual.sub_(alpha * applied)
            next_squared = torch.sum(residual * residual)
            relative_residual = float(torch.sqrt(next_squared / initial).item())
            if relative_residual < tol:
                converged = True
                squared_residual = next_squared
                break
            direction = residual + (next_squared / squared_residual) * direction
            squared_residual = next_squared
    return (
        solution.detach().cpu().numpy(),
        converged,
        iteration,
        relative_residual,
        {"linear_system_coefficient": coefficient},
    )


def solve_graph_laplacian(
    z_d: ArrayLike,
    rho: ArrayLike,
    edge_index: ArrayLike,
    omega: ArrayLike,
    s_delta: float,
    lambda_: float,
    *,
    device: str = "cpu",
    max_iters: int = 1000,
    tol: float = 1e-10,
) -> RecoveryResult:
    """Solve the matched canonical Graph-Laplacian SPD system by CG."""

    z, fidelity, edges, weights = _validated_inputs(z_d, rho, edge_index, omega, s_delta, lambda_)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if max_iters < 1 or tol <= 0:
        raise ValueError("solver iteration and tolerance settings must be positive")
    objective = lambda u: compute_graph_laplacian_objective(u, z, fidelity, edges, weights, s_delta, lambda_)
    if edges.shape[0] == 0:
        return _shortcut_result(z, objective, "conjugate_gradient_graph_laplacian", device, "no_edges")
    if lambda_ == 0:
        return _shortcut_result(z, objective, "conjugate_gradient_graph_laplacian", device, "zero_lambda")

    initial_objective = objective(z)
    coefficient = lambda_ * z.shape[0] / (edges.shape[0] * s_delta**2)
    started = perf_counter()
    backend = _gl_numpy if device == "cpu" else _gl_torch_cuda
    latent, converged, iterations, relative_change, diagnostics = backend(
        z, fidelity, edges, weights, coefficient, max_iters, tol
    )
    final_objective = objective(latent)
    diagnostics.update(
        {
            "initial_objective": initial_objective,
            "objective_change": final_objective - initial_objective,
            "elapsed_seconds": perf_counter() - started,
            "n_nodes": z.shape[0],
            "n_edges": edges.shape[0],
        }
    )
    return RecoveryResult(
        latent,
        converged,
        iterations,
        final_objective,
        relative_change,
        "conjugate_gradient_graph_laplacian",
        device,
        diagnostics,
    )
