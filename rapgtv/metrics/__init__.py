"""Canonical controlled and real-data metrics."""

from .controlled import adjusted_rand_index, boundary_auprc, nrmse_u
from .real import compute_fdd
from .spatial import EvaluationGraph, build_evaluation_graph, compute_fi

__all__ = [
    "EvaluationGraph",
    "adjusted_rand_index",
    "boundary_auprc",
    "build_evaluation_graph",
    "compute_fdd",
    "compute_fi",
    "nrmse_u",
]
