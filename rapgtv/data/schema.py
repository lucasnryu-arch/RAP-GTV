"""Canonical site-level data contract and node-order validation."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "site-dataset/v1"
_CORE_NODE_FIELDS = (
    "displacement",
    "original_valid_mask",
    "coords_projected",
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "tri",
)


def _readonly_array(value: Any, *, dtype: Any | None = None) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class SiteDataset:
    """Uniform, immutable input for every canonical site.

    ``alignment_ids`` is deliberately mandatory. Each node-wise field supplies
    the stable IDs that accompanied it at ingestion; validation requires that
    sequence to equal ``point_ids``. Keys are core field names,
    ``coords_lonlat`` when present, and ``quality_metadata.<name>`` for every
    quality array. This prevents equal-shaped but independently permuted raw
    arrays from being accepted silently.

    ``original_valid_mask`` records raw measurement existence only. It is
    copied and made read-only and is never inferred from a filled displacement.
    """

    site_id: str
    point_ids: NDArray[Any]
    times: NDArray[Any]
    displacement: NDArray[np.floating]
    original_valid_mask: NDArray[np.bool_]
    coords_projected: NDArray[np.floating]
    coords_lonlat: NDArray[np.floating] | None
    quality_metadata: Mapping[str, NDArray[Any]]
    elevation: NDArray[np.floating]
    slope: NDArray[np.floating]
    aspect: NDArray[np.floating]
    curvature: NDArray[np.floating]
    tri: NDArray[np.floating]
    optical_source: object | None
    crs_projected: str
    crs_geographic: str | None
    displacement_unit: str
    projected_coordinate_unit: str
    alignment_ids: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        array_fields = (
            "point_ids",
            "times",
            "displacement",
            "original_valid_mask",
            "coords_projected",
            "elevation",
            "slope",
            "aspect",
            "curvature",
            "tri",
        )
        for name in array_fields:
            dtype = bool if name == "original_valid_mask" else None
            object.__setattr__(self, name, _readonly_array(getattr(self, name), dtype=dtype))
        if self.coords_lonlat is not None:
            object.__setattr__(self, "coords_lonlat", _readonly_array(self.coords_lonlat))

        quality = {name: _readonly_array(value) for name, value in self.quality_metadata.items()}
        alignments = {name: _readonly_array(value) for name, value in self.alignment_ids.items()}
        object.__setattr__(self, "quality_metadata", _freeze_mapping(quality))
        object.__setattr__(self, "alignment_ids", _freeze_mapping(alignments))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        validate_site_dataset(self)

    @property
    def n_points(self) -> int:
        """Number of nodes in canonical point order."""

        return int(self.point_ids.shape[0])

    @property
    def n_times(self) -> int:
        """Number of chronological deformation epochs."""

        return int(self.times.shape[0])

    def with_displacement(self, displacement: NDArray[Any]) -> "SiteDataset":
        """Return a filled/modified value copy without changing raw validity.

        The original mask object is defensively copied again by the constructor,
        so neither the source nor result can mutate the other's provenance.
        """

        return replace(self, displacement=displacement)


def _require_text(value: str | None, name: str) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty declared value")


def _validate_time_axis(times: NDArray[Any]) -> None:
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if times.dtype.kind not in "mMif":
        raise TypeError("times must use datetime, timedelta, integer, or floating dtype")
    if times.dtype.kind == "f" and not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    if not np.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing with no duplicates")


def validate_site_dataset(dataset: SiteDataset) -> None:
    """Raise if a site violates canonical shape, identity, or metadata rules."""

    _require_text(dataset.site_id, "site_id")
    _require_text(dataset.crs_projected, "crs_projected")
    _require_text(dataset.displacement_unit, "displacement_unit")
    _require_text(dataset.projected_coordinate_unit, "projected_coordinate_unit")
    if dataset.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {dataset.schema_version!r}")

    ids = dataset.point_ids
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("point_ids must be a non-empty one-dimensional array")
    try:
        unique_count = np.unique(ids).size
    except TypeError as exc:
        raise TypeError("point_ids must contain scalar, mutually comparable stable IDs") from exc
    if unique_count != ids.size:
        raise ValueError("point_ids must be unique")
    n = ids.size

    _validate_time_axis(dataset.times)
    t = dataset.times.size
    if dataset.displacement.shape != (n, t):
        raise ValueError(f"displacement must have shape ({n}, {t})")
    if dataset.displacement.dtype.kind not in "fiu":
        raise TypeError("displacement must be numeric")
    if dataset.original_valid_mask.shape != (n, t):
        raise ValueError(f"original_valid_mask must have shape ({n}, {t})")
    if not np.all(np.isfinite(dataset.displacement[dataset.original_valid_mask])):
        raise ValueError("originally valid displacement measurements must be finite")

    if dataset.coords_projected.shape != (n, 2):
        raise ValueError(f"coords_projected must have shape ({n}, 2)")
    if not np.all(np.isfinite(dataset.coords_projected)):
        raise ValueError("coords_projected must be finite")
    if dataset.coords_lonlat is not None:
        if dataset.coords_lonlat.shape != (n, 2):
            raise ValueError(f"coords_lonlat must have shape ({n}, 2)")
        if not np.all(np.isfinite(dataset.coords_lonlat)):
            raise ValueError("coords_lonlat must be finite when supplied")
        _require_text(dataset.crs_geographic, "crs_geographic")

    for name in ("elevation", "slope", "aspect", "curvature", "tri"):
        array = getattr(dataset, name)
        if array.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},)")
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite numeric data")

    required_alignment_keys = set(_CORE_NODE_FIELDS)
    if dataset.coords_lonlat is not None:
        required_alignment_keys.add("coords_lonlat")
    for name, array in dataset.quality_metadata.items():
        if not name or not isinstance(name, str):
            raise ValueError("quality metadata names must be non-empty strings")
        if array.ndim == 0 or array.shape[0] != n:
            raise ValueError(f"quality_metadata.{name} first dimension must equal N={n}")
        required_alignment_keys.add(f"quality_metadata.{name}")

    actual_keys = set(dataset.alignment_ids)
    if actual_keys != required_alignment_keys:
        missing = sorted(required_alignment_keys - actual_keys)
        extra = sorted(actual_keys - required_alignment_keys)
        raise ValueError(f"alignment_ids keys mismatch; missing={missing}, extra={extra}")
    for name, aligned_ids in dataset.alignment_ids.items():
        if aligned_ids.shape != ids.shape or not np.array_equal(aligned_ids, ids):
            raise ValueError(f"node ordering mismatch for {name}")

