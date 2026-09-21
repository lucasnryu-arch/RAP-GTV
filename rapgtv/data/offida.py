"""Canonical adapter for the inventory-defined Offida EGMS Basic site."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from rapgtv.data.adapters import (
    ROOT,
    QualityFieldAudit,
    audit_dict,
    build_site_dataset,
    deterministic_limit,
    external_optical_audit,
    materialize_train_prefix,
    train_prefix_length,
    value_scope_audit,
    ValueScope,
)
from rapgtv.data.terrain_io import derive_and_sample_terrain


PRODUCT = "EGMS_L2a_022_0820_IW2_VV_2019_2023_1.zip"
IFFI_ID = "440370100"


def _normalize_inventory_id(value: object) -> str:
    return str(value).strip().removesuffix(".0")


def load_offida(
    *, data_root: Path | str = ROOT, max_points: int | None = None,
    value_scope: ValueScope = "all",
):
    root = Path(data_root)
    archive_path = root / PRODUCT
    inventory_path = root / "Offida_IFFI_polygons.geojson"
    inventory = gpd.read_file(inventory_path)
    selected_inventory = inventory[
        inventory["IDFRANA"].map(_normalize_inventory_id) == IFFI_ID
    ]
    if len(selected_inventory) != 1:
        raise ValueError(f"expected one IFFI polygon with IDFRANA={IFFI_ID}")
    polygon = selected_inventory.to_crs("EPSG:32633").geometry.iloc[0]
    region = polygon.buffer(200.0)

    with ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        loc_parts = []
        with archive.open(member) as stream:
            for chunk in pd.read_csv(
                stream,
                usecols=["pid", "latitude", "longitude", "temporal_coherence", "rmse"],
                chunksize=100_000,
            ):
                loc_parts.append(chunk)
        loc = pd.concat(loc_parts, ignore_index=True)
        transform = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        x, y = transform.transform(loc["longitude"].to_numpy(), loc["latitude"].to_numpy())
        points = gpd.GeoSeries(gpd.points_from_xy(x, y), crs="EPSG:32633")
        selected_rows = np.flatnonzero(points.intersects(region).to_numpy())
        selected_set = set(int(i) for i in selected_rows)

        with archive.open(member) as stream:
            header = pd.read_csv(stream, nrows=0)
        date_cols = [c for c in header.columns if re.fullmatch(r"\d{8}", str(c))]
        read_date_cols = (
            date_cols[:train_prefix_length(len(date_cols))]
            if value_scope == "train_only" else date_cols
        )
        fixed = [
            "pid", "latitude", "longitude", "rmse", "temporal_coherence",
            "amplitude_dispersion", "mean_velocity_std",
        ]
        with archive.open(member) as stream:
            frame = pd.read_csv(
                stream,
                usecols=fixed + read_date_cols,
                skiprows=lambda line: line > 0 and (line - 1) not in selected_set,
            )

    if frame.empty:
        raise RuntimeError("Offida selected IFFI polygon has no EGMS support")
    frame = frame.copy()
    x, y = transform.transform(frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    frame["_x"] = x
    frame["_y"] = y
    frame["_gx"] = np.floor(frame["_x"] / 20.0).astype(np.int64)
    frame["_gy"] = np.floor(frame["_y"] / 20.0).astype(np.int64)
    frame = (
        frame.sort_values(
            ["_gx", "_gy", "temporal_coherence", "pid"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["_gx", "_gy"])
        .reset_index(drop=True)
    )

    ids_all = frame["pid"].astype(str).to_numpy()
    order = deterministic_limit(ids_all, max_points)
    ids = ids_all[order]
    times = np.asarray([np.datetime64(c[:4] + "-" + c[4:6] + "-" + c[6:]) for c in date_cols])
    displacement = frame.loc[:, read_date_cols].to_numpy(dtype=float)[order]
    valid = np.isfinite(displacement)
    if value_scope == "train_only":
        displacement, valid = materialize_train_prefix(displacement, valid, len(date_cols))
    projected = frame.loc[:, ["_x", "_y"]].to_numpy(dtype=float)[order]
    lonlat = frame.loc[:, ["longitude", "latitude"]].to_numpy(dtype=float)[order]

    dem = next(root.glob("DEM1_*fq0n*.zip"))
    terrain, terrain_provenance = derive_and_sample_terrain(dem, lonlat, tile_hint="N42_00_E013_00")
    quality_audit = (
        QualityFieldAudit("temporal_coherence", "ACCEPTED", "higher_is_better", "EGMS temporal coherence"),
        QualityFieldAudit("rmse", "ACCEPTED", "lower_is_better", "EGMS time-series RMSE in mm"),
        QualityFieldAudit("mean_velocity_std", "ACCEPTED", "lower_is_better", "EGMS mean-velocity uncertainty"),
        QualityFieldAudit("amplitude_dispersion", "ACCEPTED", "lower_is_better", "EGMS amplitude-dispersion quality indicator"),
        QualityFieldAudit("incidence_angle/track_angle/LOS vector", "REJECTED", None, "observation geometry"),
        QualityFieldAudit("cluster_label/mp_type", "REJECTED", None, "source classification, not directed measurement quality"),
    )
    quality = {
        "temporal_coherence": frame["temporal_coherence"].to_numpy(dtype=float)[order],
        "rmse": frame["rmse"].to_numpy(dtype=float)[order],
        "mean_velocity_std": frame["mean_velocity_std"].to_numpy(dtype=float)[order],
        "amplitude_dispersion": frame["amplitude_dispersion"].to_numpy(dtype=float)[order],
    }
    optical = external_optical_audit(root, "Offida")
    inv_row = selected_inventory.iloc[0]
    metadata = {
        "deformation_observable": "EGMS Basic descending-orbit LOS displacement time series",
        "sign_convention": "EGMS LOS convention: positive toward satellite, negative away",
        "source_product": PRODUCT.removesuffix(".zip"),
        "provenance": {
            "archive": str(archive_path.resolve()),
            "member": member,
            "product_specification": "EGMS Product Description v3",
            "window": [str(times[0]), str(times[-1])],
            "raw_polygon_buffer_support": int(len(selected_rows)),
            "deduplication": "20 m projected grid; max temporal_coherence; pid tie-break",
        },
        "quality_metadata_audit": audit_dict(quality_audit),
        "terrain_provenance": terrain_provenance,
        "inventory_audit": {
            "used": True,
            "role": "site_definition_only; not emitted as a scientific input or label",
            "stable_selector": {"field": "IDFRANA", "value": IFFI_ID},
            "buffer_m": 200.0,
            "TIPOLOGIA": str(inv_row["TIPOLOGIA"]),
            "NOME_STATO": str(inv_row["NOME_STATO"]),
        },
        "optical_audit": optical,
        "raw_missing_fraction": float(1.0 - valid.mean()),
        "point_id_definition": "source EGMS pid after deterministic 20 m thinning, lexicographically ordered",
        "computational_ready": True,
        "publication_ready": True,
        "physical_metadata_status": "confirmed; inventory-informed site definition disclosed",
        "fdd_unit": "millimetre",
        "deformation_value_access": value_scope_audit(value_scope, len(date_cols)),
    }
    return build_site_dataset(
        site_id="offida",
        point_ids=ids,
        times=times,
        displacement=displacement,
        valid=valid,
        coords_projected=projected,
        coords_lonlat=lonlat,
        quality=quality,
        terrain=terrain,
        optical_source=optical,
        crs_projected="EPSG:32633",
        displacement_unit="millimetre",
        metadata=metadata,
    )
