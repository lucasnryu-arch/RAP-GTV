"""Xiongba adapter with Train-derived common source-support handling."""

from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer

from rapgtv.data.adapters import (
    ROOT,
    DataDecisionRequired,
    OpticalSourceAudit,
    QualityFieldAudit,
    audit_dict,
    build_site_dataset,
    deterministic_limit,
    materialize_train_prefix,
    train_prefix_length,
    value_scope_audit,
    ValueScope,
)
from rapgtv.data.terrain_io import derive_and_sample_terrain


Observable = Literal["ve", "vn", "vu", "horizontal_downslope"]

def _train_source_support_mask(
    files: dict[str, dict[str, Path]],
    common_dates: list[str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Derive the audited source footprint from Train-period source support.

    The invalid footprint is a fixed, cross-component source-export support mask:
    the same cells are exact zero in VE, VN, and VU at every Train epoch.  A zero
    at an individual epoch or in an individual component remains a valid value.
    """

    n_train = train_prefix_length(len(common_dates))
    persistent_common_zero = np.ones(shape, dtype=bool)
    union_common_zero = np.zeros(shape, dtype=bool)
    reference_common_zero: np.ndarray | None = None
    fixed_across_train = True
    components_equal = True
    for date in common_dates[:n_train]:
        zeros: dict[str, np.ndarray] = {}
        for component in ("ve", "vn", "vu"):
            with rasterio.open(files[component][date]) as src:
                grid = src.read(1)
            zeros[component] = grid == 0
        components_equal &= bool(
            np.array_equal(zeros["ve"], zeros["vn"])
            and np.array_equal(zeros["ve"], zeros["vu"])
        )
        common_zero = zeros["ve"] & zeros["vn"] & zeros["vu"]
        if reference_common_zero is None:
            reference_common_zero = common_zero.copy()
        else:
            fixed_across_train &= bool(np.array_equal(reference_common_zero, common_zero))
        persistent_common_zero &= common_zero
        union_common_zero |= common_zero

    if not components_equal or not fixed_across_train:
        raise RuntimeError(
            "Xiongba Train source-support footprint is not fixed across components/epochs"
        )
    if not np.array_equal(persistent_common_zero, union_common_zero):
        raise RuntimeError("Xiongba Train source-support derivation is temporally ambiguous")
    source_valid = ~persistent_common_zero
    return source_valid, {
        "mask_derivation": "TRAIN_SOURCE_SUPPORT",
        "source_support_semantics": (
            "complement of the fixed cross-component VE/VN/VU source-export footprint "
            "shared by every Train epoch; not a generic non-zero validity rule"
        ),
        "train_epoch_count": n_train,
        "invalid_raster_cells": int(persistent_common_zero.sum()),
        "source_valid_raster_cells": int(source_valid.sum()),
        "component_masks_equal": components_equal,
        "mask_fixed_across_train": fixed_across_train,
        "single_epoch_or_component_zero_invalid": False,
    }


def _dated_files(directory: Path) -> dict[str, Path]:
    result = {}
    for path in directory.glob("*.tif"):
        match = re.search(r"(\d{8})", path.name)
        if match:
            result[match.group(1)] = path
    return result


def _optical_audit(root: Path) -> OpticalSourceAudit:
    base = root / "LRPGC_data" / "xiongba" / "optical"
    monthly = tuple(str(p.resolve()) for p in sorted((base / "monthly").glob("*.tif")))
    manifest = base / "metadata" / "download_manifest.csv"
    intervals = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            intervals = [(row["start"], row["end"]) for row in csv.DictReader(stream)]
    provenance_files = [
        manifest,
        base / "metadata" / "download_config.json",
        base / "metadata" / "band_definition.txt",
        root / "download_xiongba_sentinel2_monthly.py",
    ]
    paths = monthly + tuple(str(p.resolve()) for p in provenance_files if p.exists())
    return OpticalSourceAudit(
        status="NONCANONICAL_OPTICAL_SOURCE",
        source_paths=paths,
        composite_intervals=tuple(intervals),
        acquisition_times=(),
        acquisition_ids=(),
        provenance_complete=False,
        validity_rule="exclude SCL {0,1,2,3,8,9,10,11}; monthly median; q_optical=valid_count/total_count",
        reason=(
            "monthly Process API rasters retain aggregate validity counts but not Sentinel "
            "product IDs or source-scene acquisition times"
        ),
        requires_redownload=True,
    )


def load_xiongba(
    *,
    data_root: Path | str = ROOT,
    observable: Observable = "horizontal_downslope",
    displacement_unit: str | None = "UNKNOWN — source GeoTIFFs do not declare a unit",
    sign_convention: str | None = "UNKNOWN — source component-positive convention is undocumented",
    max_points: int | None = None,
    value_scope: ValueScope = "all",
):
    if observable not in ("ve", "vn", "vu", "horizontal_downslope"):
        raise ValueError(f"unsupported Xiongba observable: {observable!r}")
    if not displacement_unit or not sign_convention:
        raise DataDecisionRequired("Xiongba unit/sign overrides must be non-empty when supplied")

    root = Path(data_root)
    base = root / "dataset" / "2018-2022" / "3d_result" / "results_cumulative"
    files = {component: _dated_files(base / component) for component in ("ve", "vn", "vu")}
    common_dates = sorted(set(files["ve"]) & set(files["vn"]) & set(files["vu"]))
    common_dates = [d for d in common_dates if "20190101" <= d <= "20220522"]
    if not common_dates:
        raise FileNotFoundError("no common Xiongba VE/VN/VU epochs in canonical window")

    with rasterio.open(files["ve"][common_dates[0]]) as reference:
        height, width = reference.height, reference.width
        transform = reference.transform
        source_crs = reference.crs
        rows_grid, cols_grid = np.indices((height, width))
        xs, ys = rasterio.transform.xy(transform, rows_grid, cols_grid, offset="center")
        x_grid = np.asarray(xs).reshape(height, width)
        y_grid = np.asarray(ys).reshape(height, width)

    source_valid_mask, source_support_audit = _train_source_support_mask(
        files, common_dates, (height, width)
    )

    if source_crs is None:
        raise ValueError("Xiongba deformation raster has no CRS")
    to_lonlat = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = to_lonlat.transform(x_grid, y_grid)
    aoi_path = root / "dataset" / "shp" / "xbm1.shp"
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326").geometry.union_all()
    points = gpd.GeoSeries(gpd.points_from_xy(np.ravel(lon_grid), np.ravel(lat_grid)), crs="EPSG:4326")
    inside = points.intersects(aoi).to_numpy().reshape(height, width)

    support_dates = (
        common_dates[:train_prefix_length(len(common_dates))]
        if value_scope == "train_only" else common_dates
    )
    valid_count = np.zeros((height, width), dtype=np.int32)
    for date in support_dates:
        jointly_valid = np.ones((height, width), dtype=bool)
        for component in ("ve", "vn", "vu"):
            with rasterio.open(files[component][date]) as src:
                values = src.read(1).astype(float)
                ok = np.isfinite(values)
                if src.nodata is not None:
                    ok &= ~np.isclose(values, src.nodata)
                jointly_valid &= ok
        valid_count += jointly_valid
    support = (
        inside
        & source_valid_mask
        & (valid_count >= int(np.ceil(0.8 * len(support_dates))))
    )
    selected_rows, selected_cols = np.nonzero(support)
    all_ids = np.asarray([f"xiongba:r{r:04d}:c{c:04d}" for r, c in zip(selected_rows, selected_cols)])
    order = deterministic_limit(all_ids, max_points)
    selected_rows, selected_cols, ids = selected_rows[order], selected_cols[order], all_ids[order]
    lonlat = np.column_stack((lon_grid[selected_rows, selected_cols], lat_grid[selected_rows, selected_cols]))
    to_projected = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    px, py = to_projected.transform(lonlat[:, 0], lonlat[:, 1])
    projected = np.column_stack((px, py))
    terrain, terrain_provenance = derive_and_sample_terrain(
        root / "dataset" / "shp" / "dem.tif", lonlat
    )

    read_dates = (
        common_dates[:train_prefix_length(len(common_dates))]
        if value_scope == "train_only" else common_dates
    )
    displacement = np.full((len(ids), len(read_dates)), np.nan, dtype=float)
    valid = np.zeros_like(displacement, dtype=bool)
    for j, date in enumerate(read_dates):
        sampled = {}
        sampled_valid = {}
        for component in ("ve", "vn", "vu"):
            with rasterio.open(files[component][date]) as src:
                grid = src.read(1).astype(float)
                values = grid[selected_rows, selected_cols]
                ok = np.isfinite(values)
                if src.nodata is not None:
                    ok &= ~np.isclose(values, src.nodata)
                sampled[component] = values
                sampled_valid[component] = ok
        if observable == "horizontal_downslope":
            angle = np.deg2rad(terrain["aspect"])
            displacement[:, j] = sampled["ve"] * np.sin(angle) + sampled["vn"] * np.cos(angle)
            valid[:, j] = sampled_valid["ve"] & sampled_valid["vn"]
        else:
            displacement[:, j] = sampled[observable]
            valid[:, j] = sampled_valid[observable]
        displacement[~valid[:, j], j] = np.nan
    if value_scope == "train_only":
        displacement, valid = materialize_train_prefix(displacement, valid, len(common_dates))

    optical = _optical_audit(root)
    quality_audit = (
        QualityFieldAudit("full-period ascending coherence", "REJECTED", None, "time-support overlaps future construction periods"),
        QualityFieldAudit("full-period descending coherence", "REJECTED", None, "time-support overlaps future construction periods"),
        QualityFieldAudit("incidence/aspect rasters", "REJECTED", None, "geometry, not directed measurement quality"),
    )
    metadata = {
        "deformation_observable": observable,
        "sign_convention": sign_convention,
        "source_product": "Xiongba 3-D cumulative displacement VE/VN/VU GeoTIFF time series",
        "provenance": {
            "root": str(base.resolve()),
            "dataset_doi": "https://doi.org/10.17632/pss9p5p2n5.1",
            "component_mapping": {"ve": "east", "vn": "north", "vu": "vertical/up"},
            "window": ["2019-01-01", "2022-05-22"],
            "support_rule": (
                "inside xbm1 AOI, Train-derived source-support valid, and jointly "
                "finite VE/VN/VU at >=80% of the requested support-period epochs"
            ),
            "source_support_audit": source_support_audit,
        },
        "quality_metadata_audit": audit_dict(quality_audit),
        "terrain_provenance": terrain_provenance,
        "inventory_audit": {"used": False, "role": "none; xbm1 is an AOI, not an inventory label"},
        "optical_audit": optical,
        "raw_missing_fraction": float(1.0 - valid.mean()),
        "point_id_definition": "source raster row/column in the fixed deformation grid",
        "computational_ready": True,
        "publication_ready": False,
        "physical_metadata_status": "pending: raw unit and component-positive sign convention",
        "fdd_unit": "native_unknown",
        "observable_audit": {
            "status": (
                "PUBLICATION_DEFAULT"
                if observable == "horizontal_downslope"
                else "EXPLICIT_NONDEFAULT_OVERRIDE"
            ),
            "alternatives": ["ve", "vn", "vu", "horizontal_downslope"],
            "canonical_default": "horizontal_downslope",
            "projection_formula": "VE * sin(aspect) + VN * cos(aspect)",
            "aspect_definition": (
                "clockwise from geographic north; 0 degrees=north, 90 degrees=east; "
                "terrain downslope direction"
            ),
            "legacy_habit": (
                "the legacy loader independently uses the same horizontal-downslope projection"
            ),
            "source_component_meanings": {
                "ve": "east component",
                "vn": "north component",
                "vu": "vertical/up component",
            },
            "component_meaning_evidence": (
                "source directory labels plus the legacy loader's explicit east/north use; "
                "GeoTIFF band descriptions are empty"
            ),
            "source_unit_status": "UNKNOWN: GeoTIFF unit tag and band description are empty",
            "source_sign_status": "UNKNOWN: no local product document defines component-positive directions",
        },
        "deformation_value_access": {
            **value_scope_audit(value_scope, len(common_dates)),
            "support_epoch_count": len(support_dates),
        },
    }
    return build_site_dataset(
        site_id="xiongba",
        point_ids=ids,
        times=np.asarray([np.datetime64(d[:4] + "-" + d[4:6] + "-" + d[6:]) for d in common_dates]),
        displacement=displacement,
        valid=valid,
        coords_projected=projected,
        coords_lonlat=lonlat,
        quality={},
        terrain=terrain,
        optical_source=optical,
        crs_projected="EPSG:32647",
        displacement_unit=displacement_unit,
        metadata=metadata,
    )
