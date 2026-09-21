# Reproduction guide

## Lightweight verification

```bash
python -m pip install -e ".[test]"
python examples/minimal_example.py
pytest -q
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py --output-dir outputs/paper_figures
```

These commands require no private data. They verify the implementation and regenerate the quantitative table and result-based figures from versioned reference values.

## Full controlled design

```bash
python scripts/run_controlled.py --full --output-dir outputs/controlled_full
```

The full design uses 20 seeds and the conditions in `configs/controlled.json`. Runtime depends on the solver backend and hardware.

## Full real-site runs

Install the data and optical extras, arrange the input files as documented, and call `scripts/run_rapgtv.py`. Run one seed at a time using the site configuration. The manuscript summaries use seeds 0–4.

Site maps and deformation-history panels require non-redistributed source products. The repository therefore supplies their computation path but cannot regenerate them from a fresh clone alone.
