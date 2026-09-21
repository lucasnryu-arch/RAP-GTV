# Data layout

No real-site source data are distributed in this repository. Create the ignored directories below and place legally obtained files in the stated locations.

## Xiongba

- Type: 3-D cumulative InSAR displacement components, DEM, site polygon, and training-period Sentinel-2 summaries.
- Provenance: dataset DOI `10.17632/pss9p5p2n5.1` is recorded by the adapter; redistribution rights for the local source package were not confirmed.
- Redistributed here: no.
- Expected layout:

```text
data/raw/xiongba/
  dataset/2018-2022/3d_result/results_cumulative/{ve,vn,vu}/*.tif
  dataset/shp/xbm1.*
  dataset/shp/dem.tif
  optical/provenance.json
  optical/monthly_indices.nc
```

The adapter derives a common source-support mask from the training-period VE/VN/VU rasters. An isolated zero at one epoch or in one component remains a valid measurement; only the fixed common source-export footprint is excluded.

## Offida

- Type: EGMS L2a descending line-of-sight displacement, IFFI polygon, DEM, and training-period Sentinel-2 summaries.
- Original sources: European Ground Motion Service and Italian Landslide Inventory.
- Redistribution status: not confirmed for this assembled local package.
- Redistributed here: no.
- Expected layout:

```text
data/raw/offida/
  EGMS_L2a_022_0820_IW2_VV_2019_2023_1.zip
  Offida_IFFI_polygons.geojson
  DEM1_*fq0n*.zip
  optical/provenance.json
  optical/monthly_indices.nc
```

## Mendatica

- Type: descending InSAR displacement archive, DEM, and training-period Sentinel-2 summaries.
- Original source: research data archive used by the manuscript; exact redistribution permission was not confirmed.
- Redistributed here: no.
- Expected layout:

```text
data/raw/mendatica/
  Mendatica landslide research data.zip
  DEM1_*hx1f*.zip
  optical/provenance.json
  optical/monthly_indices.nc
```

## Zhouqu–Xieliupo

- Type: SBAS-InSAR cumulative line-of-sight point product, DEM, site polygon embedded in the archive, and training-period Sentinel-2 summaries.
- Original source: provider archive used by the manuscript; exact redistribution permission was not confirmed.
- Redistributed here: no.
- Expected layout:

```text
data/raw/zhouqu_xieliupo/
  2yy6z3sfvr-1.zip
  DEM1_*magC*.zip
  optical/provenance.json
  optical/monthly_indices.nc
```

## Tiny example data

`examples/minimal_example.py` generates all of its input arrays deterministically at runtime. It contains no sample from any real site.
