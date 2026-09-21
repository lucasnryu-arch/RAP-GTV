"""Cutoff-safe optical representation, reliability, and edge gate."""

from .core import (
    OpticalCompositeSeries,
    OpticalGate,
    OpticalRepresentation,
    build_optical_representation,
    compute_optical_gate,
    compute_optical_indices,
)


def __getattr__(name: str):
    """Load optional xarray-backed provenance helpers only when requested."""

    if name in {"load_openeo_source_audit", "load_openeo_composites"}:
        from .provenance import load_openeo_composites, load_openeo_source_audit

        return {
            "load_openeo_source_audit": load_openeo_source_audit,
            "load_openeo_composites": load_openeo_composites,
        }[name]
    raise AttributeError(name)

__all__ = [
    "OpticalCompositeSeries",
    "OpticalGate",
    "OpticalRepresentation",
    "build_optical_representation",
    "compute_optical_gate",
    "compute_optical_indices",
    "load_openeo_source_audit",
    "load_openeo_composites",
]
