"""Canonical adapter for the Zhouqu–Xieliupo SBAS-InSAR point product."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
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


def _stable_id(fid: int, lon: float, lat: float) -> str:
    identity = f"SBAS-InSAR monitoring results around Xieliupo Landslide|{fid}|{lon:.9f}|{lat:.9f}"
    return "zhouqu:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def load_zhouqu(
    *, data_root: Path | str = ROOT, max_points: int | None = None,
    value_scope: ValueScope = "all",
):
    root = Path(data_root)
    outer_path = root / "2yy6z3sfvr-1.zip"
    with TemporaryDirectory(prefix="rapgtv-zhouqu-") as temp:
        temp_path = Path(temp)
        with ZipFile(outer_path) as outer:
            nested_name = next(n for n in outer.namelist() if n.endswith("SBAS-InSAR monitoring results around Xieliupo Landslide.zip"))
            nested_path = temp_path / "xieliupo.zip"
            nested_path.write_bytes(outer.read(nested_name))
        with ZipFile(nested_path) as nested:
            nested.extractall(temp_path / "shape")
        shp = next((temp_path / "shape").rglob("*.shp"))
        import pyogrio

        source_info = pyogrio.read_info(shp)
        source_crs = source_info["crs"]
        source_fields = list(source_info["fields"])
        date_cols = [c for c in source_fields if c.startswith("D_") and len(c) == 10]
        times_all = np.asarray([np.datetime64(c[2:6] + "-" + c[6:8] + "-" + c[8:10]) for c in date_cols])
        keep_t = (times_all >= np.datetime64("2015-07-04")) & (times_all <= np.datetime64("2020-02-16"))
        date_cols = list(np.asarray(date_cols)[keep_t])
        times = times_all[keep_t]
        read_date_cols = (
            date_cols[:train_prefix_length(len(date_cols))]
            if value_scope == "train_only" else date_cols
        )
        columns = ["Coherence", *read_date_cols]
        frame = gpd.read_file(shp, columns=columns)

    if source_crs is None:
        raise ValueError("Zhouqu shapefile has no declared CRS")
    frame = frame.set_crs(source_crs, allow_override=True)
    frame = frame.to_crs("EPSG:4326")
    lonlat_all = np.column_stack((frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()))
    raw_fids = np.arange(len(frame), dtype=int)
    ids_all = np.asarray([_stable_id(int(i), x, y) for i, (x, y) in zip(raw_fids, lonlat_all)], dtype=str)
    order = deterministic_limit(ids_all, max_points)
    ids = ids_all[order]
    lonlat = lonlat_all[order]

    displacement = frame.loc[:, read_date_cols].to_numpy(dtype=float)[order]
    valid = np.isfinite(displacement)
    if value_scope == "train_only":
        displacement, valid = materialize_train_prefix(displacement, valid, len(date_cols))

    transform = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    x, y = transform.transform(lonlat[:, 0], lonlat[:, 1])
    projected = np.column_stack((x, y))
    dem = next(root.glob("DEM1_*magC*.zip"))
    terrain, terrain_provenance = derive_and_sample_terrain(dem, lonlat, tile_hint="N33_00_E104_00")
    quality_audit = (
        QualityFieldAudit("Coherence", "ACCEPTED", "higher_is_better", "standard interferometric coherence"),
        QualityFieldAudit("H_Precisio", "UNRESOLVED", None, "field semantics and direction are not documented in the archive"),
        QualityFieldAudit("V_Precisio", "UNRESOLVED", None, "field semantics and direction are not documented in the archive"),
        QualityFieldAudit("L1Norm", "REJECTED", None, "fit statistic is not a documented node-quality observable"),
        QualityFieldAudit("ChiSqr/ChiSqr_1", "REJECTED", None, "undocumented fit statistics"),
    )
    quality = {"coherence": frame["Coherence"].to_numpy(dtype=float)[order]}
    optical = external_optical_audit(root, "Zhouqu")
    metadata = {
        "deformation_observable": "source SBAS-InSAR cumulative LOS displacement relative to D_20150513",
        "sign_convention": "UNKNOWN — not encoded in archive metadata; decision required before physical interpretation",
        "source_product": "SBAS-InSAR monitoring results around Xieliupo Landslide shapefile",
        "provenance": {
            "archive": str(outer_path.resolve()),
            "nested_member": nested_name,
            "source_crs": str(frame.crs),
            "window": ["2015-07-04", "2020-02-16"],
            "source_date_range": ["2015-05-13", "2020-02-16"],
            "reference_epoch": "2015-05-13 (D_20150513 is exactly zero for all 27,713 source features)",
            "temporal_semantics": "cumulative/relative displacement series, not per-interval increments",
        },
        "quality_metadata_audit": audit_dict(quality_audit),
        "terrain_provenance": terrain_provenance,
        "inventory_audit": {"used": False, "role": "none"},
        "optical_audit": optical,
        "raw_missing_fraction": float(1.0 - valid.mean()),
        "point_id_definition": "SHA-256(nested product identity, original feature index, canonical lon/lat), first 20 hex",
        "computational_ready": True,
        "publication_ready": False,
        "physical_metadata_status": "pending: unit, LOS sign, and orbit direction",
        "fdd_unit": "native_unknown",
        "data_decisions_required": ["confirm displacement unit", "confirm LOS sign convention"],
        "deformation_value_access": value_scope_audit(value_scope, len(date_cols)),
    }
    return build_site_dataset(
        site_id="zhouqu-xieliupo",
        point_ids=ids,
        times=times,
        displacement=displacement,
        valid=valid,
        coords_projected=projected,
        coords_lonlat=lonlat,
        quality=quality,
        terrain=terrain,
        optical_source=optical,
        crs_projected="EPSG:32648",
        displacement_unit="UNKNOWN — source archive does not declare a unit",
        metadata=metadata,
    )
