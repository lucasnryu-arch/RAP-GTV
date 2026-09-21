"""Causal deformation representation primitives."""

from .deformation import (
    DeformationRepresentation,
    build_deformation_representation,
    build_working_displacement,
    compute_time_aware_rate,
    long_term_statistics,
)
from .paa import paa
from .scaling import RobustScaler

__all__ = [
    "DeformationRepresentation",
    "RobustScaler",
    "build_deformation_representation",
    "build_working_displacement",
    "compute_time_aware_rate",
    "long_term_statistics",
    "paa",
]
