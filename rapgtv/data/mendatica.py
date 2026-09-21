"""Canonical adapter for the Mendatica EGMS L2B research dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

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


def load_mendatica(
    *, data_root: Path | str = ROOT, max_points: int | None = None,
    value_scope: ValueScope = "all",
):
    root = Path(data_root)
    archive_path = root / "Mendatica landslide research data.zip"
    with ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith("PS/SNT_desc.csv"))
        header = pd.read_csv(archive.open(member), sep=";", nrows=0)

    date_cols = [c for c in header if c.startswith("D") and len(c) == 9]
    parsed = np.array([np.datetime64(datetime.strptime(c[1:], "%d%m%Y")) for c in date_cols])
    keep_t = (parsed >= np.datetime64("2015-07-04")) & (parsed <= np.datetime64("2019-12-25"))
    date_cols = list(np.asarray(date_cols)[keep_t])
    times = parsed[keep_t]
    read_date_cols = (
        date_cols[:train_prefix_length(len(date_cols))]
        if value_scope == "train_only" else date_cols
    )
    fixed = ["pid", "easting", "northing", "coherence", "vel_std"]
    with ZipFile(archive_path) as archive:
        frame = pd.read_csv(archive.open(member), sep=";", usecols=fixed + read_date_cols)

    ids = frame["pid"].astype(str).to_numpy()
    order = deterministic_limit(ids, max_points)
    ids = ids[order]
    displacement = frame.loc[:, read_date_cols].to_numpy(dtype=float)[order]
    valid = np.isfinite(displacement)
    if value_scope == "train_only":
        displacement, valid = materialize_train_prefix(displacement, valid, len(date_cols))
    coords_source = frame.loc[:, ["easting", "northing"]].to_numpy(dtype=float)[order]
    to_projected = Transformer.from_crs("EPSG:3035", "EPSG:32632", always_xy=True)
    to_lonlat = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    px, py = to_projected.transform(coords_source[:, 0], coords_source[:, 1])
    lon, lat = to_lonlat.transform(coords_source[:, 0], coords_source[:, 1])
    projected = np.column_stack((px, py))
    lonlat = np.column_stack((lon, lat))

    dem = next(root.glob("DEM1_*hx1f*.zip"))
    terrain, terrain_provenance = derive_and_sample_terrain(dem, lonlat, tile_hint="N44_00_E007_00")
    quality_audit = (
        QualityFieldAudit("coherence", "ACCEPTED", "higher_is_better", "documented InSAR coherence"),
        QualityFieldAudit("vel_std", "ACCEPTED", "lower_is_better", "documented velocity uncertainty"),
        QualityFieldAudit("eff_area", "REJECTED", None, "point-type attribute, not a directed quality score"),
        QualityFieldAudit("inc_angle", "REJECTED", None, "observation geometry"),
        QualityFieldAudit("track_angl", "REJECTED", None, "observation geometry"),
        QualityFieldAudit("los_east/los_north/los_up", "REJECTED", None, "LOS unit-vector geometry"),
    )
    quality = {
        "coherence": frame["coherence"].to_numpy(dtype=float)[order],
        "vel_std": frame["vel_std"].to_numpy(dtype=float)[order],
    }
    optical = external_optical_audit(root, "Mendatica")
    metadata = {
        "deformation_observable": "Sentinel-1 descending-orbit LOS displacement",
        "sign_convention": "positive toward satellite; negative away from satellite",
        "source_product": "EGMS L2B SNT descending, relative orbit 66",
        "provenance": {
            "archive": str(archive_path.resolve()),
            "member": member,
            "dataset_doi": "https://doi.org/10.17632/3k8t43xs2x.1",
            "source_crs": "EPSG:3035",
            "date_columns": "Dddmmyyyy",
            "window": ["2015-07-04", "2019-12-25"],
        },
        "quality_metadata_audit": audit_dict(quality_audit),
        "terrain_provenance": terrain_provenance,
        "inventory_audit": {"used": False, "role": "none"},
        "optical_audit": optical,
        "raw_missing_fraction": float(1.0 - valid.mean()),
        "point_id_definition": "source EGMS pid, lexicographically ordered",
        "computational_ready": True,
        "publication_ready": True,
        "physical_metadata_status": "confirmed",
        "fdd_unit": "millimetre",
        "deformation_value_access": value_scope_audit(value_scope, len(date_cols)),
    }
    return build_site_dataset(
        site_id="mendatica",
        point_ids=ids,
        times=times,
        displacement=displacement,
        valid=valid,
        coords_projected=projected,
        coords_lonlat=lonlat,
        quality=quality,
        terrain=terrain,
        optical_source=optical,
        crs_projected="EPSG:32632",
        displacement_unit="millimetre",
        metadata=metadata,
    )
