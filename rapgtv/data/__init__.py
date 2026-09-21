"""Canonical data contracts."""

from .schema import SiteDataset, validate_site_dataset

__all__ = ["SiteDataset", "validate_site_dataset"]
"""Canonical data contract and real-data adapters."""

from rapgtv.data.schema import SiteDataset

__all__ = ["SiteDataset"]
