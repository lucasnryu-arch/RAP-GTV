# Third-party software and comparison methods

No upstream comparison-method source tree is vendored in this repository.

| Method or package | Upstream and citation | License finding | Included here | User action |
|---|---|---|---|---|
| Time2Feat-KM | Time2Feat repository `softlab-unimore/time2feat`, reference commit `ad247463d0204f182cb9b8ef2b51dc804b29d5dc`; tsfresh: Christ et al., *Neurocomputing* 307, 72–77 (2018), DOI `10.1016/j.neucom.2018.03.067` | No upstream source is copied. tsfresh is an external dependency distributed under its own terms. | Project-authored adapter using tsfresh descriptors, deterministic feature reduction, and K-means. | Install the `baselines` optional dependencies. |
| K-SC | Yang and Leskovec, “Patterns of Temporal Variation in Online Media,” WSDM 2011; Stanford source page `https://snap.stanford.edu/data/ksc.html`; landslide application: Liang et al., *Engineering Geology* 357, 108367 (2025), DOI `10.1016/j.enggeo.2025.108367` | The source page supplies a MATLAB archive, but archive-specific redistribution terms were not sufficiently explicit for this release. | No K-SC source. A label-file adapter and exact settings are documented in `baselines/README.md`. | Obtain and run an authorized implementation separately, then evaluate its aligned labels. |
| RAC-DMVC | `https://github.com/LouisDong95/RAC-DMVC`, reference commit `441fafcae958a586f5d8d9727069be4104825035`; Dong et al., AAAI 40(25), 20835–20843 (2026), DOI `10.1609/aaai.v40i25.39223` | The inspected upstream repository did not expose a recognized license file. Upstream source is not copied. | Project-authored three-view adaptation used by the manuscript. | Install PyTorch separately for a full run. |
| NumPy, SciPy, pandas, matplotlib, GeoPandas, pyproj, rasterio, xarray, h5py, h5netcdf, PyTorch, pytest | Project package indexes and upstream repositories | Each remains under its own license. | Dependencies only. | Review dependency licenses for redistribution of packaged environments. |

The project-authored adapters are part of RAP-GTV and remain subject to the repository's pending project-license decision.
