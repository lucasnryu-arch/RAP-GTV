"""Canonical RAP-GTV and matched Graph-Laplacian solvers."""

from .recovery import (
    FORMAL_DUAL_FEASIBILITY_TOL,
    FORMAL_RELATIVE_PRIMAL_DUAL_GAP_TOL,
    RAPGTVCertificate,
    RecoveryResult,
    compute_graph_laplacian_objective,
    compute_rap_gtv_certificate,
    compute_rap_gtv_objective,
    solve_graph_laplacian,
    solve_rap_gtv,
)

__all__ = [
    "FORMAL_DUAL_FEASIBILITY_TOL",
    "FORMAL_RELATIVE_PRIMAL_DUAL_GAP_TOL",
    "RAPGTVCertificate",
    "RecoveryResult",
    "compute_graph_laplacian_objective",
    "compute_rap_gtv_certificate",
    "compute_rap_gtv_objective",
    "solve_graph_laplacian",
    "solve_rap_gtv",
]
