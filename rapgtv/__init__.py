"""RAP-GTV scientific implementation."""

__version__ = "0.9.0"

from .data.schema import SiteDataset, validate_site_dataset
from .graph.weights import compute_edge_conductance, compute_jump_scale
from .representation.deformation import build_deformation_representation
from .temporal.split import TemporalSplit, build_temporal_split

__all__ = [
    "__version__",
    "SiteDataset",
    "TemporalSplit",
    "build_deformation_representation",
    "build_temporal_split",
    "compute_edge_conductance",
    "compute_jump_scale",
    "validate_site_dataset",
]
