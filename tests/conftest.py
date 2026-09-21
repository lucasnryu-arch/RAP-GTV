from __future__ import annotations

import numpy as np

from rapgtv.data.schema import SiteDataset


def make_site_dataset(n: int = 12, t: int = 40) -> SiteDataset:
    ids = np.asarray([f"P{i:03d}" for i in range(n)])
    times = np.datetime64("2020-01-01") + np.arange(t) * np.timedelta64(12, "D")
    displacement = (np.arange(n)[:, None] * 0.2 + np.arange(t)[None, :] * 0.05).astype(float)
    valid = np.ones((n, t), dtype=bool)
    valid[1, 3] = False
    displacement[~valid] = np.nan
    coords = np.column_stack((np.arange(n) * 20.0, np.arange(n) * 7.0))
    quality = {"coherence": np.linspace(0.7, 0.95, n)}
    names = {
        "displacement",
        "original_valid_mask",
        "coords_projected",
        "elevation",
        "slope",
        "aspect",
        "curvature",
        "tri",
        "quality_metadata.coherence",
    }
    return SiteDataset(
        site_id="test-site",
        point_ids=ids,
        times=times,
        displacement=displacement,
        original_valid_mask=valid,
        coords_projected=coords,
        coords_lonlat=None,
        quality_metadata=quality,
        elevation=1200.0 + np.arange(n),
        slope=np.linspace(5.0, 35.0, n),
        aspect=np.linspace(0.0, 315.0, n),
        curvature=np.linspace(-0.2, 0.2, n),
        tri=np.linspace(1.0, 4.0, n),
        optical_source=None,
        crs_projected="synthetic-grid",
        crs_geographic=None,
        displacement_unit="mm",
        projected_coordinate_unit="metre",
        alignment_ids={name: ids.copy() for name in names},
        metadata={"source": "synthetic test fixture"},
    )
