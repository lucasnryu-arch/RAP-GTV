# Comparison methods

## Time2Feat-KM

`rapgtv.baselines.time2feat` implements the manuscript pipeline: comprehensive tsfresh descriptors, removal of non-finite and constant features, standardization, SVD retaining 90% cumulative variance, clustering of feature loadings, one representative per loading cluster, re-standardization, and final K-means with the shared site `K*`.

## K-SC

K-SC source is not distributed here. Obtain an authorized implementation from the original authors or the Stanford source page and use these settings:

- signed training-period deformation series;
- no K-SC-specific rebasing, centring, or normalization;
- integer shifts from -5 through +5 with zero padding;
- optimized pairwise scale for every shift;
- one initialization for each seed;
- stop when memberships are unchanged;
- maximum 1000 iterations;
- shared site `K*` from the RAP-GTV configuration.

Save the resulting integer labels as a one-dimensional NumPy `.npy` file in the same point order as the site adapter. Validate and evaluate them with `scripts/evaluate.py`.

## RAC-DMVC

`rapgtv.baselines.rac_dmvc` contains the project-authored three-view adaptation used for the manuscript. Its fixed settings are: hidden widths `[1024, 1024, 1024]`, latent dimension 128, dropout 0.2, noise ratio 0.5, sigma 0.07, contrastive and distillation temperatures 0.5, 100 epochs, 20 warm-up epochs, rectification from epoch 20, batch size 1024, base learning rate `5e-4`, EMA momentum 0.98, and no early stopping.

See `docs/third_party.md` for provenance, citations, and license findings.
