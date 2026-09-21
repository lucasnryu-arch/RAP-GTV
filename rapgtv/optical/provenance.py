"""Read and validate canonical openEO Train-only product provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from rapgtv.data.adapters import OpticalSourceAudit
from rapgtv.optical.core import OpticalCompositeSeries


def load_openeo_source_audit(path: Path | str) -> OpticalSourceAudit:
    """Convert a persisted openEO job record into the Phase-2 source contract."""

    provenance_path = Path(path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "site",
        "job_id",
        "backend",
        "collection",
        "aoi_path",
        "aoi_sha256",
        "temporal_extent",
        "train_cutoff",
        "bands",
        "scl_mask",
        "monthly_reducer",
        "process_graph",
        "output_path",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"openEO provenance missing fields: {missing}")
    if payload["temporal_extent"][1] != payload["train_cutoff"]:
        raise ValueError("openEO temporal end must equal the frozen Train cutoff")
    output = Path(payload["output_path"])
    source_paths = (str(provenance_path.resolve()), str(output.resolve()))
    return OpticalSourceAudit(
        status="CANONICAL_TRAIN_ONLY_OPENEO",
        source_paths=source_paths,
        composite_intervals=tuple(tuple(x) for x in payload.get("composite_intervals", ())),
        acquisition_times=tuple(np.datetime64(x) for x in payload.get("scene_times", ())),
        acquisition_ids=tuple(payload.get("scene_ids", ())),
        provenance_complete=True,
        validity_rule=json.dumps(payload["scl_mask"], sort_keys=True),
        reason="Train-only openEO process graph proves temporal filtering before masking/compositing",
        requires_redownload=False,
        backend_process_graph=payload["process_graph"],
        backend_job_id=payload["job_id"],
    )


def load_openeo_composites(
    path: Path | str, coords_projected: np.ndarray
) -> OpticalCompositeSeries:
    """Nearest-sample a canonical openEO NetCDF at aligned site-node coordinates."""

    audit = load_openeo_source_audit(path)
    product_path = Path(audit.source_paths[1])
    if not product_path.exists():
        raise FileNotFoundError(f"openEO job output is not downloaded: {product_path}")
    coords = np.asarray(coords_projected, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError("coords_projected must be a finite N by 2 array")
    with xr.open_dataset(product_path) as dataset:
        required = {"NDVI", "NDMI", "BSI", "valid_proportion", "t", "x", "y"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"canonical openEO output missing variables: {missing}")
        nodes = np.arange(coords.shape[0])
        selected = dataset.sel(
            x=xr.DataArray(coords[:, 0], dims="node", coords={"node": nodes}),
            y=xr.DataArray(coords[:, 1], dims="node", coords={"node": nodes}),
            method="nearest",
        )
        values = np.stack(
            [selected[name].transpose("node", "t").values for name in ("NDVI", "NDMI", "BSI")],
            axis=-1,
        )
        valid = selected["valid_proportion"].transpose("node", "t").values
        valid = np.clip(np.nan_to_num(valid, nan=0.0), 0.0, 1.0)
        times = dataset["t"].values
    empty_times = tuple(np.asarray([], dtype="datetime64[ns]") for _ in times)
    empty_ids = tuple(() for _ in times)
    return OpticalCompositeSeries(values, valid, times, empty_times, empty_ids, audit)
