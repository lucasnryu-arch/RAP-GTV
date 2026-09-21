# Data preparation

## Scope

The real-site adapters preserve point identity, chronological acquisition order, original validity, projected coordinates, and terrain fields. The repository does not download or redistribute source products.

## Procedure

1. Obtain the site source files from the original provider or authors under the applicable terms.
2. Create the site directory described in `data/README.md`.
3. Obtain Copernicus DEM and training-period Sentinel-2 imagery from official Copernicus services. Do not place large rasters under version control.
4. Produce monthly NDVI, NDMI, and BSI composites and a `provenance.json` record compatible with `rapgtv.optical.provenance.load_openeo_composites`.
5. Validate the site adapter:

```bash
python scripts/prepare_data.py --site <site> --data-root data/raw/<site>
```

The expected output is `data/processed/<site>/dataset_summary.json`. It records shapes, units, the chronological split, and stable point-identity digest but does not copy source observations.

## Optical input contract

The NetCDF product must contain aligned monthly values for the optical bands used to derive NDVI, NDMI, and BSI, plus valid-observation proportions. Source-scene acquisition times or a verifiable backend process graph must demonstrate that only training-period observations enter construction.

## Xiongba support rule

For every training epoch, the adapter compares zero-valued cells across VE, VN, and VU. A cell is outside source support only when the same zero footprint is fixed across all three components and every training epoch. Per-epoch zeros and single-component zeros are retained. This rule yields the 35,711 points used in the manuscript when applied to the corresponding source package and site polygon.
