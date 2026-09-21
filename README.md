# RAP-GTV

Reliability-Aware Physical Graph Total Variation for Landslide Deformation Zoning

Official implementation accompanying the manuscript by Shuo Lu and Hui He:

“Landslide deformation zoning through reliability-aware InSAR deformation-field recovery with terrain and optical constraints.”

## Overview

RAP-GTV converts irregular InSAR time series into spatially coherent deformation regimes while retaining local deformation changes. Observation quality determines fidelity to each measurement; terrain and optical observations constrain only the graph interactions.

```text
InSAR time series
  -> deformation representation
  -> observation reliability
  -> terrain-aware graph
  -> optical edge modulation
  -> vector Graph-TV recovery
  -> deformation regimes
```

The repository contains the scientific implementation, site adapters, final experiment settings, lightweight verification tests, and machine-readable values reported in the manuscript. It does not redistribute the real-site source datasets.

## Repository structure

- `rapgtv/`: deformation representation, reliability, graph construction, optical modulation, solver, zoning, metrics, and site adapters.
- `configs/`: controlled and four-site settings reported in the manuscript.
- `scripts/`: data checks, experiment entry points, evaluation, tables, and figures.
- `reproduce/`: ordered reproduction entry points and limitations.
- `paper_results/`: machine-readable reference values for the manuscript.
- `tests/`: small CPU tests of science-critical behavior.
- `docs/`: data, method-to-code, availability, and third-party documentation.

## Installation

Python 3.11 is the tested version.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For the real-site adapters and optical products:

```bash
python -m pip install -e ".[data,optical,baselines,figures]"
```

PyTorch is optional. Full RAC-DMVC runs and CUDA-accelerated RAP-GTV runs require a PyTorch installation appropriate for the host system. The minimal example and tests run on CPU.

An equivalent Conda environment is provided in `environment.yml`.

## Quick start

This self-contained synthetic example exercises deformation representation, reliability, graph construction, optical modulation, Graph-TV recovery, and zoning without private data or a GPU:

```bash
python examples/minimal_example.py
```

It prints the node count, selected regime count, solver convergence, and runtime. It is a smoke test, not a reproduction of a paper result.

## Data preparation

Place source files under `data/raw/<site>/` according to `data/README.md`, then check one site with:

```bash
python scripts/prepare_data.py --site offida --data-root data/raw/offida
```

The command validates the adapter inputs and writes only a compact dataset summary under `data/processed/<site>/`. Optical monthly composites and their provenance records are prepared separately as described in `docs/data_preparation.md`.

## Controlled experiments

Run a small controlled check:

```bash
python scripts/run_controlled.py --quick --output-dir outputs/controlled_quick
```

The complete controlled design is available with `--full`. It is computationally larger and is not part of CI.

## Real-site experiments

After obtaining the corresponding source dataset and training-period optical product:

```bash
python scripts/run_rapgtv.py --site offida --data-root data/raw/offida --optical-provenance data/raw/offida/optical/provenance.json --output-dir outputs/offida
```

Full real-site experiments benefit from CUDA, but the same solver supports CPU execution. Site-specific reproduction requires the source datasets and can be substantially more expensive than the quick start.

## Comparison methods

Project-authored Time2Feat-KM and RAC-DMVC adapters are available through:

```bash
python scripts/run_baselines.py --help
```

K-SC source is not redistributed because the upstream archive's redistribution terms were not sufficiently explicit for this release. `baselines/README.md` records the source, settings, citation, and external-label interface.

## Evaluation

Evaluate labels produced by RAP-GTV or an external comparison method:

```bash
python scripts/evaluate.py --dataset-npz data/processed/site/dataset.npz --labels labels.npy --output metrics.json
```

FDD is evaluated on the chronological test period. FI always uses the coordinate-only symmetric 8-nearest-neighbour evaluation graph.

## Reproducing paper tables

Table 2 can be regenerated entirely from the versioned reference values:

```bash
python scripts/make_paper_tables.py
```

The generated CSV is checked numerically against `paper_results/table2_real_benchmark.csv`.

## Reproducing paper figures

The quantitative controlled and four-site comparison panels can be generated from the versioned result CSV files:

```bash
python scripts/make_paper_figures.py --output-dir outputs/paper_figures
```

Maps and trajectory panels that depend on non-redistributed real-site rasters require the corresponding source data. Input requirements and coverage are listed in `reproduce/README.md`.

## Data availability

This repository does not redistribute Xiongba, Offida, Mendatica, Zhouqu–Xieliupo, Sentinel-2, EGMS, or DEM source products. Obtain each dataset from its original provider or the authors according to its terms. See `data/README.md` and `docs/data_availability.md` for the exact local layout and known provenance.

## Code availability

The repository provides the computational pipeline, experiment settings, evaluation code, and scripts for the principal quantitative tables and figures. Reproduction of site-specific results requires the corresponding source datasets.

## Citation

Software citation metadata are provided in `CITATION.cff`. Manuscript citation information will be updated after publication; no paper or archival DOI is currently asserted.

## License

This repository is released under the MIT License. See `LICENSE` for details.

Third-party provenance and redistribution decisions are documented in `docs/third_party.md`.
