# Paper result references

These files contain machine-readable reference values reported in the manuscript.

- `table2_real_benchmark.csv`: the 4 sites × 4 methods shown in Table 2, including means, sample standard deviations, and within-site ranks.
- `controlled_summary.csv`: controlled-experiment summaries used by the quantitative controlled figure.
- `generated_table2.csv`: output produced by `scripts/make_paper_tables.py`; it must match the reference table numerically.

FDD values are compared only within site. Xiongba and Zhouqu–Xieliupo retain the source unit label `native_unknown`; the Italian sites use millimetres. FI is computed from the coordinate-only symmetric 8-nearest-neighbour evaluation graph. Algorithm seeds characterize computational variability and are not independent physical samples.
