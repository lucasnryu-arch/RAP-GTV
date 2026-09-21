"""Shared, fail-closed machinery for the four Phase-3A real-data adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from calendar import monthrange
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from rapgtv.audit.leakage import LeakageAudit, LeakageError, audit_acquisition_times
from rapgtv.data.schema import SiteDataset
from rapgtv.temporal.causality import ConstructionContext


ROOT = Path(__file__).resolve().parents[3]
QualityStatus = Literal["ACCEPTED", "REJECTED", "UNRESOLVED"]
ValueScope = Literal["all", "train_only"]


class DataDecisionRequired(RuntimeError):
    """A scientifically material choice is not encoded by the source data."""


def train_prefix_length(n_epochs: int) -> int:
    """Return the frozen 60% Train prefix length without inspecting observations."""

    if n_epochs < 3:
        raise ValueError("at least three epochs are required for the frozen split")
    return int(np.floor(0.60 * n_epochs))


def materialize_train_prefix(
    train_values: np.ndarray,
    train_valid: np.ndarray,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place Train values in a full-axis array whose later periods are unread placeholders."""

    values = np.asarray(train_values, dtype=float)
    valid = np.asarray(train_valid, dtype=bool)
    n_train = train_prefix_length(n_epochs)
    if values.ndim != 2 or values.shape[1] != n_train or valid.shape != values.shape:
        raise ValueError("Train materialization does not match the frozen split prefix")
    result = np.full((values.shape[0], n_epochs), np.nan, dtype=float)
    mask = np.zeros((values.shape[0], n_epochs), dtype=bool)
    result[:, :n_train] = values
    mask[:, :n_train] = valid
    return result, mask


def value_scope_audit(value_scope: ValueScope, n_epochs: int) -> Mapping[str, Any]:
    """Describe the adapter-level observation firewall in machine-auditable form."""

    n_train = train_prefix_length(n_epochs)
    return {
        "value_scope": value_scope,
        "full_epoch_count": int(n_epochs),
        "materialized_value_indices": [0, n_train - 1] if value_scope == "train_only" else [0, n_epochs - 1],
        "validation_values_materialized": value_scope != "train_only",
        "test_values_materialized": value_scope != "train_only",
        "post_train_placeholders": "NaN with original_valid_mask=false" if value_scope == "train_only" else None,
    }


@dataclass(frozen=True, slots=True)
class QualityFieldAudit:
    field: str
    status: QualityStatus
    direction: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class OpticalSourceAudit:
    status: str
    source_paths: tuple[str, ...]
    composite_intervals: tuple[tuple[str, str], ...]
    acquisition_times: tuple[np.datetime64, ...]
    acquisition_ids: tuple[str, ...]
    provenance_complete: bool
    validity_rule: str
    reason: str
    requires_redownload: bool
    backend_process_graph: Mapping[str, Any] | None = None
    backend_job_id: str | None = None


def _graph_ancestors(graph: Mapping[str, Any], node_id: str) -> set[str]:
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            parent = value.get("from_node")
            if isinstance(parent, str) and parent in graph and parent not in seen:
                seen.add(parent)
                visit(graph[parent].get("arguments", {}))
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(graph[node_id].get("arguments", {}))
    return seen


def validate_backend_process_graph(
    graph: Mapping[str, Any], context: ConstructionContext
) -> LeakageAudit:
    """Accept backend proof only when filtering and masking precede aggregation."""

    loads = {key: node for key, node in graph.items() if node.get("process_id") == "load_collection"}
    aggregates = {
        key: node for key, node in graph.items() if node.get("process_id") == "aggregate_temporal_period"
    }
    masks = {key for key, node in graph.items() if node.get("process_id") == "mask"}
    mask_generators = {
        key for key, node in graph.items() if node.get("process_id") == "to_scl_dilation_mask"
    }
    if not loads or not aggregates or not masks:
        raise LeakageError("backend process graph lacks load_collection, mask, or monthly aggregation")
    checked_times = []
    for node in loads.values():
        extent = node.get("arguments", {}).get("temporal_extent")
        if not isinstance(extent, list) or len(extent) != 2 or not all(extent):
            raise LeakageError("load_collection has no closed temporal_extent")
        checked_times.extend(np.asarray(extent, dtype="datetime64[ns]"))
    try:
        context.assert_times_allowed(np.asarray(checked_times), source="optical backend temporal filters")
    except ValueError as exc:
        raise LeakageError(str(exc)) from exc
    has_masked_composite = False
    for node_id in aggregates:
        ancestors = _graph_ancestors(graph, node_id)
        if not (ancestors & set(loads)):
            raise LeakageError("monthly aggregation is not downstream of temporal filtering")
        if ancestors & masks:
            has_masked_composite = True
        elif not (ancestors & mask_generators):
            raise LeakageError("monthly aggregation is not downstream of masking and temporal filtering")
    if not has_masked_composite:
        raise LeakageError("backend process graph has no masked monthly science composite")
    return LeakageAudit(
        context.stage,
        "optical backend process graph",
        context.cutoff_time,
        len(checked_times),
    )


def validate_optical_source(
    source: OpticalSourceAudit, context: ConstructionContext
) -> LeakageAudit:
    """Require acquisition-level provenance, then enforce the construction cutoff."""

    if not source.provenance_complete:
        raise LeakageError(
            "optical source has composites but no complete source-scene acquisition IDs/times"
        )
    if not source.acquisition_times and source.backend_process_graph is not None:
        if not source.backend_job_id:
            raise LeakageError("backend process graph provenance has no job ID")
        return validate_backend_process_graph(source.backend_process_graph, context)
    if len(source.acquisition_times) != len(source.acquisition_ids) or not source.acquisition_times:
        raise LeakageError("optical acquisition IDs/times and backend proof are absent or incomplete")
    return audit_acquisition_times(
        context,
        np.asarray(source.acquisition_times, dtype="datetime64[ns]"),
        source="optical source-scene acquisition",
    )


def audit_dict(items: Sequence[QualityFieldAudit]) -> tuple[Mapping[str, Any], ...]:
    return tuple(asdict(item) for item in items)


def build_site_dataset(
    *,
    site_id: str,
    point_ids: np.ndarray,
    times: np.ndarray,
    displacement: np.ndarray,
    valid: np.ndarray,
    coords_projected: np.ndarray,
    coords_lonlat: np.ndarray,
    quality: Mapping[str, np.ndarray],
    terrain: Mapping[str, np.ndarray],
    optical_source: OpticalSourceAudit,
    crs_projected: str,
    displacement_unit: str,
    metadata: Mapping[str, Any],
) -> SiteDataset:
    ids = np.asarray(point_ids, dtype=str)
    keys = {
        "displacement",
        "original_valid_mask",
        "coords_projected",
        "coords_lonlat",
        "elevation",
        "slope",
        "aspect",
        "curvature",
        "tri",
    }
    keys.update(f"quality_metadata.{name}" for name in quality)
    alignments = {key: ids for key in keys}
    required = {
        "deformation_observable",
        "sign_convention",
        "source_product",
        "provenance",
        "quality_metadata_audit",
        "terrain_provenance",
        "inventory_audit",
        "optical_audit",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"adapter metadata missing required audit fields: {missing}")
    return SiteDataset(
        site_id=site_id,
        point_ids=ids,
        times=np.asarray(times, dtype="datetime64[ns]"),
        displacement=np.asarray(displacement, dtype=float),
        original_valid_mask=np.asarray(valid, dtype=bool),
        coords_projected=np.asarray(coords_projected, dtype=float),
        coords_lonlat=np.asarray(coords_lonlat, dtype=float),
        quality_metadata=quality,
        elevation=terrain["elevation"],
        slope=terrain["slope"],
        aspect=terrain["aspect"],
        curvature=terrain["curvature"],
        tri=terrain["tri"],
        optical_source=optical_source,
        crs_projected=crs_projected,
        crs_geographic="EPSG:4326",
        displacement_unit=displacement_unit,
        projected_coordinate_unit="metre",
        alignment_ids=alignments,
        metadata=metadata,
    )


def deterministic_limit(ids: np.ndarray, max_points: int | None) -> np.ndarray:
    order = np.argsort(np.asarray(ids, dtype=str), kind="stable")
    if max_points is not None:
        if max_points < 1:
            raise ValueError("max_points must be positive")
        order = order[:max_points]
    return order


def external_optical_audit(root: Path, site: str) -> OpticalSourceAudit:
    directory_name = "Zhouqu_Xieliupo" if site == "Zhouqu" else site
    base = root / "RAPGTV_Experiment3_Sentinel2" / directory_name
    data_paths = list(sorted(base.rglob("*.nc"))) if base.exists() else []
    provenance_paths = [
        root / "RAPGTV_Experiment3_Sentinel2" / "Sentinel2_Audit.csv",
        root / "RAPGTV_Experiment3_Sentinel2" / "Sentinel2_Download_Log.csv",
        root / "RAPGTV_Experiment3_Sentinel2" / "job_registry.json",
        root / "Experiment3_Sentinel2_FINAL_v2.0_ASCII_SAFE_RESUME.py",
    ]
    paths = tuple(str(p.resolve()) for p in data_paths + provenance_paths if p.exists())
    windows = {
        "Mendatica": (date(2015, 7, 4), date(2019, 12, 25)),
        "Zhouqu": (date(2015, 7, 4), date(2020, 2, 16)),
        "Offida": (date(2019, 1, 1), date(2023, 12, 31)),
    }
    lower, upper = windows[site]
    intervals = []
    cursor = date(lower.year, lower.month, 1)
    while cursor <= upper:
        month_end = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        intervals.append((max(cursor, lower).isoformat(), min(month_end, upper).isoformat()))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return OpticalSourceAudit(
        status="NONCANONICAL_OPTICAL_SOURCE",
        source_paths=paths,
        composite_intervals=tuple(intervals),
        acquisition_times=(),
        acquisition_ids=(),
        provenance_complete=False,
        validity_rule=(
            "CDSE openEO to_scl_dilation_mask: kernel1=17, kernel2=77, "
            "mask1_values={2,4,5,6,7}, mask2_values={3,8,9,10,11}, "
            "erosion_kernel_size=3; then monthly median"
        ),
        reason=(
            "stored NetCDF files are server-side monthly composites; openEO job IDs are "
            "available but Sentinel product IDs and source-scene acquisition times were "
            "not retained, so a construction cutoff cannot be audited"
        ),
        requires_redownload=True,
    )
