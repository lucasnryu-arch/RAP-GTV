"""Common Copernicus/GeoTIFF terrain derivation used by every site adapter."""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import numpy as np


@contextmanager
def _open_dem(path: Path, tile_hint: str | None = None) -> Iterator[object]:
    import rasterio

    if path.suffix.lower() != ".zip":
        with rasterio.open(path) as src:
            yield src
        return
    with ZipFile(path) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if tile_hint:
            candidates = [n for n in candidates if tile_hint in n] or candidates
        if not candidates:
            raise FileNotFoundError(f"no GeoTIFF in DEM archive {path}")
        with rasterio.MemoryFile(archive.read(sorted(candidates)[0])) as mem:
            with mem.open() as src:
                yield src


def derive_and_sample_terrain(
    dem_path: Path,
    lonlat: np.ndarray,
    *,
    tile_hint: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Derive one fixed terrain definition and nearest-sample it at point locations.

    Slope and aspect are in degrees; aspect is downslope clockwise from north.
    Curvature is the elevation Laplacian (1/m), and TRI is the 3x3 population
    standard deviation of elevation (m). Geographic pixel sizes are converted
    locally to metres before derivatives are taken.
    """

    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt, uniform_filter

    with _open_dem(Path(dem_path), tile_hint) as src:
        z = src.read(1).astype(float)
        invalid = ~np.isfinite(z)
        if src.nodata is not None:
            invalid |= np.isclose(z, src.nodata)
        fill_count = int(invalid.sum())
        if invalid.all():
            raise ValueError(f"DEM contains no finite cells: {dem_path}")
        if invalid.any():
            nearest = distance_transform_edt(invalid, return_distances=False, return_indices=True)
            z = z[tuple(nearest)]

        bounds = src.bounds
        mid_lat = (bounds.bottom + bounds.top) / 2 if src.crs and src.crs.is_geographic else 0.0
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        if src.crs and src.crs.is_geographic:
            dx *= 111_320.0 * np.cos(np.deg2rad(mid_lat))
            dy *= 110_574.0
        gy, gx = np.gradient(z, dy, dx)
        gyy = np.gradient(gy, dy, axis=0)
        gxx = np.gradient(gx, dx, axis=1)
        slope_grid = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect_grid = (np.degrees(np.arctan2(-gx, -gy)) + 360.0) % 360.0
        curvature_grid = gxx + gyy
        mean = uniform_filter(z, size=3, mode="nearest")
        mean_sq = uniform_filter(z * z, size=3, mode="nearest")
        tri_grid = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lonlat[:, 0], lonlat[:, 1])
        from rasterio.transform import rowcol

        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        if np.any(rows < 0) or np.any(rows >= src.height) or np.any(cols < 0) or np.any(cols >= src.width):
            raise ValueError(f"site point falls outside DEM extent: {dem_path}")
        terrain = {
            "elevation": z[rows, cols],
            "slope": slope_grid[rows, cols],
            "aspect": aspect_grid[rows, cols],
            "curvature": curvature_grid[rows, cols],
            "tri": tri_grid[rows, cols],
        }
        provenance = {
            "source": str(Path(dem_path).resolve()),
            "tile_hint": tile_hint,
            "source_crs": str(src.crs),
            "sampling": "nearest DEM cell after EPSG:4326 coordinate transform",
            "nodata_handling": "nearest finite DEM cell before derivation",
            "nodata_cells_filled": fill_count,
            "slope": "degrees(arctan(hypot(dz/dx,dz/dy)))",
            "aspect": "degrees clockwise from north, downslope",
            "curvature": "d2z/dx2 + d2z/dy2, unit 1/m",
            "tri": "3x3 population standard deviation of elevation, unit m",
            "derivative_spacing_metres": [float(dx), float(dy)],
        }
        return terrain, provenance
